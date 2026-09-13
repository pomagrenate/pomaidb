// pomai/pulp.cc — Fast approximate representation (Pulp) implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "pulp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace pomai::storage {

PulpBuilder::PulpBuilder(uint32_t dim, uint8_t quant_type)
    : dim_(dim), count_(0), quant_type_(quant_type),
      min_val_(0.0f), scale_(0.0f), inv_scale_(0.0f) {}

void PulpBuilder::Train(const std::vector<std::span<const float>>& vectors) {
    if (vectors.empty() || dim_ == 0) return;

    float g_min = std::numeric_limits<float>::max();
    float g_max = std::numeric_limits<float>::lowest();
    bool found_valid = false;

    for (const auto& v : vectors) {
        if (v.size() != dim_) continue;
        for (float f : v) {
            if (std::isfinite(f)) {
                if (f < g_min) g_min = f;
                if (f > g_max) g_max = f;
                found_valid = true;
            }
        }
    }

    if (!found_valid) {
        min_val_ = 0.0f;
        scale_ = 0.0f;
        inv_scale_ = 0.0f;
        return;
    }

    min_val_ = g_min;
    const float range = g_max - g_min;
    if (range <= 1e-6f) {
        scale_ = 0.0f;
        inv_scale_ = 0.0f;
    } else {
        scale_ = 255.0f / range;
        inv_scale_ = range / 255.0f;
    }
}

void PulpBuilder::EncodeAppend(std::span<const float> vec) {
    if (vec.size() != dim_) return;

    const size_t prev_size = buffer_.size();
    buffer_.resize(prev_size + dim_);
    uint8_t* dst = buffer_.data() + prev_size;

    if (scale_ <= 0.0f) {
        std::memset(dst, 0, dim_);
    } else {
        for (size_t i = 0; i < dim_; ++i) {
            float normalized = (vec[i] - min_val_) * scale_;
            float clamped = std::clamp(normalized, 0.0f, 255.0f);
            dst[i] = static_cast<uint8_t>(std::round(clamped));
        }
    }
    ++count_;
}

} // namespace pomai::storage
