// include/pvec/pvec_kmeans.h — Fast K-Means Clustering for Vector Centroid Discovery
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>
#include "pvec_distance.h"

namespace pvec {

struct KMeansConfig {
    std::size_t max_iterations{25};
    float tolerance{1e-4f};
    uint64_t seed{42};
};

class KMeans {
public:
    /// Runs Lloyd's K-Means on n vectors of dimension `dim` (row-major flat array)
    /// Returns K centroids (flat vector of size K * dim)
    static std::vector<float> Fit(const float* data,
                                  std::size_t n,
                                  std::size_t dim,
                                  std::size_t k,
                                  const KMeansConfig& cfg = {}) {
        if (!data || n == 0 || dim == 0 || k == 0) return {};
        k = std::min(k, n);

        std::vector<float> centroids(k * dim);
        std::mt19937_64 rng(cfg.seed);

        // K-Means++ Initialization
        std::uniform_int_distribution<std::size_t> uniform_dist(0, n - 1);
        std::size_t first_idx = uniform_dist(rng);
        std::copy_n(data + first_idx * dim, dim, centroids.data());

        std::vector<float> min_distances(n, 1e30f);
        for (std::size_t c = 1; c < k; ++c) {
            std::span<const float> last_centroid(centroids.data() + (c - 1) * dim, dim);
            double total_weight = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                std::span<const float> pt(data + i * dim, dim);
                float d = l2_sq(pt, last_centroid);
                min_distances[i] = std::min(min_distances[i], d);
                total_weight += static_cast<double>(min_distances[i]);
            }

            std::uniform_real_distribution<double> real_dist(0.0, total_weight);
            double target = real_dist(rng);
            double cumulative = 0.0;
            std::size_t next_idx = n - 1;
            for (std::size_t i = 0; i < n; ++i) {
                cumulative += static_cast<double>(min_distances[i]);
                if (cumulative >= target) {
                    next_idx = i;
                    break;
                }
            }
            std::copy_n(data + next_idx * dim, dim, centroids.data() + c * dim);
        }

        // Iteration
        std::vector<std::size_t> assignments(n, 0);
        std::vector<float> new_centroids(k * dim, 0.0f);
        std::vector<std::size_t> counts(k, 0);

        for (std::size_t iter = 0; iter < cfg.max_iterations; ++iter) {
            bool changed = false;

            // Assignment step
            for (std::size_t i = 0; i < n; ++i) {
                std::span<const float> pt(data + i * dim, dim);
                float best_d = 1e30f;
                std::size_t best_c = 0;
                for (std::size_t c = 0; c < k; ++c) {
                    std::span<const float> cent(centroids.data() + c * dim, dim);
                    float d = l2_sq(pt, cent);
                    if (d < best_d) {
                        best_d = d;
                        best_c = c;
                    }
                }
                if (assignments[i] != best_c) {
                    assignments[i] = best_c;
                    changed = true;
                }
            }

            if (!changed && iter > 0) break;

            // Update step
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (std::size_t i = 0; i < n; ++i) {
                std::size_t c = assignments[i];
                counts[c]++;
                const float* pt = data + i * dim;
                float* cent = new_centroids.data() + c * dim;
                for (std::size_t d = 0; d < dim; ++d) {
                    cent[d] += pt[d];
                }
            }

            float max_shift = 0.0f;
            for (std::size_t c = 0; c < k; ++c) {
                if (counts[c] == 0) continue;
                float inv_n = 1.0f / static_cast<float>(counts[c]);
                float* new_cent = new_centroids.data() + c * dim;
                for (std::size_t d = 0; d < dim; ++d) {
                    new_cent[d] *= inv_n;
                }
                float shift = l2_sq(std::span<const float>(centroids.data() + c * dim, dim),
                                    std::span<const float>(new_cent, dim));
                if (shift > max_shift) max_shift = shift;
            }

            centroids = std::move(new_centroids);
            if (max_shift < cfg.tolerance) break;
            new_centroids.resize(k * dim, 0.0f);
        }

        return centroids;
    }
};

} // namespace pvec
