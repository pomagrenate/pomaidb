// reference_distance.h — Exact, obvious scalar reference implementation (oracle)
//
// Pure standard C++ scalar loops with double-precision accumulation.
// Used as the ground-truth oracle for differential testing of SIMD kernels.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pomai::core::reference {

/// Scalar reference for L2 squared distance: sum((a_i - b_i)^2)
double L2Sq(std::span<const float> a, std::span<const float> b) noexcept;

/// Scalar reference for Euclidean L2 distance: sqrt(sum((a_i - b_i)^2))
double L2(std::span<const float> a, std::span<const float> b) noexcept;

/// Scalar reference for Inner Product (Dot Product): sum(a_i * b_i)
double InnerProduct(std::span<const float> a, std::span<const float> b) noexcept;

/// Scalar reference for Cosine Similarity: dot(a, b) / (||a|| * ||b||)
/// Invariant: returns 0.0 when ||a|| == 0, ||b|| == 0, or dim == 0.
double CosineSimilarity(std::span<const float> a, std::span<const float> b) noexcept;

/// Scalar reference for Cosine Distance: 1.0 - CosineSimilarity(a, b)
/// Range: [0.0, 2.0]. Returns 1.0 when similarity is 0.0.
double CosineDistance(std::span<const float> a, std::span<const float> b) noexcept;

/// Scalar reference for SQ8 dequantized dot product
double DotSq8(std::span<const float> query,
              std::span<const uint8_t> codes,
              float min_val, float inv_scale, float query_sum = 0.0f) noexcept;

/// Scalar reference for SQ8 dequantized L2 squared
double L2SqSq8(std::span<const float> query,
               std::span<const uint8_t> codes,
               float min_val, float max_val) noexcept;

/// Scalar reference for FP16 dot product
double DotFp16(std::span<const float> query,
               std::span<const uint16_t> codes) noexcept;

/// Scalar reference for FP16 L2 squared
double L2SqFp16(std::span<const float> query,
                std::span<const uint16_t> codes) noexcept;

/// Scalar reference for Hamming distance between bit arrays
uint32_t HammingDist(std::span<const uint8_t> a, std::span<const uint8_t> b) noexcept;

/// Scalar reference for 1-bit quantization
void BitQuantize(std::span<const float> vec, uint8_t* out_codes) noexcept;

} // namespace pomai::core::reference
