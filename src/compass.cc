// pomai/compass.cc — Compass: Spatial routing layer implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "compass.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "distance.h"
#include "pomegranate_compute.h"
#include <cstring>

namespace pomai::routing {

Compass::Compass(MetricType metric) : metric_(metric) {}

Compass::~Compass() = default;

void Compass::UpdateLocules(std::vector<alloc::SharedPtr<storage::Locule>> locules) {
    locules_ = std::move(locules);
    matrix_.count = static_cast<uint32_t>(locules_.size());
    matrix_.dim = 0;
    matrix_.data.clear();
    matrix_.radii.clear();
    matrix_.norms.clear();

    if (locules_.empty()) return;

    for (const auto& loc : locules_) {
        if (loc && !loc->anchor().centroid.empty()) {
            matrix_.dim = static_cast<uint32_t>(loc->anchor().centroid.size());
            break;
        }
    }

    if (matrix_.dim == 0) return;

    const uint32_t dim = matrix_.dim;
    const uint32_t count = matrix_.count;
    matrix_.data.resize(static_cast<size_t>(count) * dim, 0.0f);
    matrix_.radii.resize(count, 0.0f);
    matrix_.norms.resize(count, 1.0f);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& loc = locules_[i];
        if (!loc) continue;
        matrix_.radii[i] = loc->anchor().radius;
        const auto& c = loc->anchor().centroid;
        if (c.size() == dim) {
            std::memcpy(matrix_.data.data() + static_cast<size_t>(i) * dim, c.data(), dim * sizeof(float));
            if (metric_ != MetricType::kL2) {
                float norm_sq = 0.0f;
                for (float v : c) norm_sq += v * v;
                matrix_.norms[i] = std::sqrt(std::max(0.0f, norm_sq));
            }
        }
    }
}

std::vector<OrientedLocule> Compass::Orient(std::span<const float> query,
                                            uint32_t nprobe,
                                            float distance_ratio_threshold) const {
    std::vector<OrientedLocule> results;
    results.reserve(locules_.size());

    if (locules_.empty()) return results;

    float q_norm = 1.0f;
    if (metric_ != MetricType::kL2) {
        float q_norm_sq = 0.0f;
        for (float v : query) {
            q_norm_sq += v * v;
        }
        q_norm = std::sqrt(std::max(0.0f, q_norm_sq));
    }

    const uint32_t dim = matrix_.dim;
    const uint32_t count = matrix_.count;

    if (dim > 0 && dim == query.size() && count > 0) {
        const float* q_ptr = query.data();
        const float* mat_ptr = matrix_.data.data();

        uint32_t i = 0;
        for (; i + 3 < count; i += 4) {
            const float* c0 = mat_ptr + static_cast<size_t>(i + 0) * dim;
            const float* c1 = mat_ptr + static_cast<size_t>(i + 1) * dim;
            const float* c2 = mat_ptr + static_cast<size_t>(i + 2) * dim;
            const float* c3 = mat_ptr + static_cast<size_t>(i + 3) * dim;

            float d[4];
            if (metric_ == MetricType::kL2) {
                compute::L2SqF32_4x(q_ptr, c0, c1, c2, c3, dim, d);
                for (uint32_t k = 0; k < 4; ++k) {
                    if (!locules_[i + k]) continue;
                    float dist = std::sqrt(std::max(0.0f, d[k]));
                    float radius = matrix_.radii[i + k];
                    float lower_bound = std::max(0.0f, dist - radius);
                    results.push_back({locules_[i + k], dist, lower_bound * lower_bound});
                }
            } else {
                compute::DotF32_4x(q_ptr, c0, c1, c2, c3, dim, d);
                for (uint32_t k = 0; k < 4; ++k) {
                    if (!locules_[i + k]) continue;
                    float dot = d[k];
                    float radius = matrix_.radii[i + k];
                    results.push_back({locules_[i + k], dot, dot + q_norm * radius});
                }
            }
        }

        for (; i < count; ++i) {
            if (!locules_[i]) continue;
            const float* c = mat_ptr + static_cast<size_t>(i) * dim;
            float radius = matrix_.radii[i];
            if (metric_ == MetricType::kL2) {
                float l2sq = core::L2Sq(query, std::span<const float>(c, dim));
                float dist = std::sqrt(std::max(0.0f, l2sq));
                float lower_bound = std::max(0.0f, dist - radius);
                results.push_back({locules_[i], dist, lower_bound * lower_bound});
            } else {
                float dot = core::Dot(query, std::span<const float>(c, dim));
                results.push_back({locules_[i], dot, dot + q_norm * radius});
            }
        }
    } else {
        // Fallback for locules with uninitialized/varying centroid dimensions
        for (const auto& loc : locules_) {
            if (!loc) continue;

            OrientedLocule ol;
            ol.locule = loc;

            const auto& centroid = loc->anchor().centroid;
            if (centroid.empty() || centroid.size() != query.size()) {
                ol.distance_to_centroid = 0.0f;
                ol.min_possible_distance = 0.0f;
                results.push_back(std::move(ol));
                continue;
            }

            float radius = loc->anchor().radius;

            if (metric_ == MetricType::kL2) {
                float l2sq = core::L2Sq(query, centroid);
                float dist = std::sqrt(std::max(0.0f, l2sq));
                ol.distance_to_centroid = dist;
                float lower_bound = std::max(0.0f, dist - radius);
                ol.min_possible_distance = lower_bound * lower_bound;
            } else {
                float dot = core::Dot(query, centroid);
                ol.distance_to_centroid = dot;
                ol.min_possible_distance = dot + q_norm * radius;
            }

            results.push_back(std::move(ol));
        }
    }

    // Sort by proximity
    if (metric_ == MetricType::kL2) {
        std::sort(results.begin(), results.end(), [](const OrientedLocule& a, const OrientedLocule& b) {
            return a.distance_to_centroid < b.distance_to_centroid;
        });
    } else {
        std::sort(results.begin(), results.end(), [](const OrientedLocule& a, const OrientedLocule& b) {
            return a.distance_to_centroid > b.distance_to_centroid;
        });
    }

    // Adaptive early termination: prune remaining candidate clusters if centroid distance
    // exceeds a threshold relative to the closest cluster
    if (!results.empty() && distance_ratio_threshold > 0.0f) {
        float ratio = distance_ratio_threshold;
        if (ratio <= 1.0f) {
            ratio = 1.0f + ratio; // e.g. 0.25 -> 1.25x
        }
        if (metric_ == MetricType::kL2) {
            float min_dist = results[0].distance_to_centroid;
            float max_allowed = min_dist * ratio;
            size_t valid = 1;
            while (valid < results.size() && results[valid].distance_to_centroid <= max_allowed) {
                valid++;
            }
            results.resize(valid);
        } else {
            float max_sim = results[0].distance_to_centroid;
            float min_allowed = (max_sim > 0.0f) ? (max_sim / ratio) : (max_sim * ratio);
            size_t valid = 1;
            while (valid < results.size() && results[valid].distance_to_centroid >= min_allowed) {
                valid++;
            }
            results.resize(valid);
        }
    }

    if (nprobe > 0 && results.size() > nprobe) {
        results.resize(nprobe);
    }

    return results;
}

std::vector<OrientedLocule> Compass::Peel(const std::vector<OrientedLocule>& candidates,
                                         float worst_distance) const {
    if (std::isinf(worst_distance)) {
        return candidates;
    }

    std::vector<OrientedLocule> surviving;
    surviving.reserve(candidates.size());

    for (const auto& c : candidates) {
        if (metric_ == MetricType::kL2) {
            // In L2, lower is better. If min_possible_distance (squared) > worst_distance, prune.
            if (c.min_possible_distance <= worst_distance) {
                surviving.push_back(c);
            }
        } else {
            // In Dot, higher is better. If max_possible_score < worst_distance, prune.
            if (c.min_possible_distance >= worst_distance) {
                surviving.push_back(c);
            }
        }
    }

    return surviving;
}

} // namespace pomai::routing
