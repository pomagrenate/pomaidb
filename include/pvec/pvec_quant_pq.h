// include/pvec/pvec_quant_pq.h — Product Quantization (PQ) with Asymmetric Distance Computation (ADC)
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include "pvec_distance.h"
#include "pvec_kmeans.h"

namespace pvec {

struct PQConfig {
    std::size_t num_subspaces{8};      // M: e.g. 8 or 16 subspaces
    std::size_t centroids_per_subspace{256}; // K: 256 fits in 1 byte (uint8_t)
    std::size_t kmeans_iterations{15};
};

class ProductQuantizer {
public:
    ProductQuantizer() = default;

    /// Train codebooks on training vectors
    bool Train(const float* data, std::size_t n, std::size_t dim, const PQConfig& cfg = {}) {
        if (!data || n == 0 || dim == 0 || cfg.num_subspaces == 0) return false;
        if (dim % cfg.num_subspaces != 0) return false;

        m_ = cfg.num_subspaces;
        k_ = std::min(cfg.centroids_per_subspace, n);
        d_sub_ = dim / m_;
        dim_ = dim;

        // Allocate codebooks: M subspaces, each has K centroids of dimension d_sub_
        codebooks_.resize(m_ * k_ * d_sub_);

        // Subspace training buffer
        std::vector<float> sub_data(n * d_sub_);

        for (std::size_t m = 0; m < m_; ++m) {
            // Extract subspace m from all n vectors
            for (std::size_t i = 0; i < n; ++i) {
                const float* src = data + i * dim_ + m * d_sub_;
                float* dst = sub_data.data() + i * d_sub_;
                std::copy_n(src, d_sub_, dst);
            }

            KMeansConfig km_cfg;
            km_cfg.max_iterations = cfg.kmeans_iterations;
            km_cfg.seed = 42 + m * 1337;

            auto centroids = KMeans::Fit(sub_data.data(), n, d_sub_, k_, km_cfg);
            if (centroids.empty()) return false;

            std::copy(centroids.begin(), centroids.end(),
                      codebooks_.begin() + m * k_ * d_sub_);
        }

        trained_ = true;
        return true;
    }

    /// Encodes a vector into M bytes
    void Encode(std::span<const float> vec, uint8_t* out_codes) const noexcept {
        if (!trained_ || !out_codes || vec.size() != dim_) return;

        for (std::size_t m = 0; m < m_; ++m) {
            std::span<const float> sub_vec(vec.data() + m * d_sub_, d_sub_);
            const float* cb = codebooks_.data() + m * k_ * d_sub_;

            float best_d = 1e30f;
            uint8_t best_k = 0;
            for (std::size_t k = 0; k < k_; ++k) {
                std::span<const float> cent(cb + k * d_sub_, d_sub_);
                float d = l2_sq(sub_vec, cent);
                if (d < best_d) {
                    best_d = d;
                    best_k = static_cast<uint8_t>(k);
                }
            }
            out_codes[m] = best_k;
        }
    }

    /// Precompute Look-Up Table (LUT) for Asymmetric Distance Computation (ADC)
    /// lut size must be >= M * K floats
    void ComputeL2DistanceTable(std::span<const float> query, float* lut) const noexcept {
        if (!trained_ || !lut || query.size() != dim_) return;

        for (std::size_t m = 0; m < m_; ++m) {
            std::span<const float> query_sub(query.data() + m * d_sub_, d_sub_);
            const float* cb = codebooks_.data() + m * k_ * d_sub_;
            float* lut_sub = lut + m * k_;

            for (std::size_t k = 0; k < k_; ++k) {
                std::span<const float> cent(cb + k * d_sub_, d_sub_);
                lut_sub[k] = l2_sq(query_sub, cent);
            }
        }
    }

    /// ADC Distance evaluation using precomputed LUT
    inline float ComputeDistanceWithTable(const float* lut, const uint8_t* codes) const noexcept {
        float dist = 0.0f;
        for (std::size_t m = 0; m < m_; ++m) {
            dist += lut[m * k_ + codes[m]];
        }
        return dist;
    }

    [[nodiscard]] bool is_trained() const noexcept { return trained_; }
    [[nodiscard]] std::size_t num_subspaces() const noexcept { return m_; }
    [[nodiscard]] std::size_t subspace_dim() const noexcept { return d_sub_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }

private:
    std::size_t dim_{0};
    std::size_t m_{0};
    std::size_t k_{0};
    std::size_t d_sub_{0};
    std::vector<float> codebooks_;
    bool trained_{false};
};

} // namespace pvec
