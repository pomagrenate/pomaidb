// pomai/compass.cc — Compass: Spatial routing layer implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "compass.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "distance.h"

namespace pomai::routing {

Compass::Compass(MetricType metric) : metric_(metric) {}

Compass::~Compass() = default;

void Compass::UpdateLocules(std::vector<alloc::SharedPtr<storage::Locule>> locules) {
    locules_ = std::move(locules);
}

std::vector<OrientedLocule> Compass::Orient(std::span<const float> query, uint32_t nprobe) const {
    std::vector<OrientedLocule> results;
    results.reserve(locules_.size());

    float q_norm = 1.0f;
    if (metric_ != MetricType::kL2) {
        float q_norm_sq = 0.0f;
        for (float v : query) {
            q_norm_sq += v * v;
        }
        q_norm = std::sqrt(std::max(0.0f, q_norm_sq));
    }

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
            ol.min_possible_distance = lower_bound * lower_bound; // L2-squared lower bound
        } else {
            // Dot or Cosine (larger is better)
            float dot = core::Dot(query, centroid);
            ol.distance_to_centroid = dot;
            // Upper bound on dot score: <q, c> + ||q|| * radius
            ol.min_possible_distance = dot + q_norm * radius;
        }

        results.push_back(std::move(ol));
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
