#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <cstddef>
#include "options.h"
#include "vector_batch.h"

namespace pomai::core
{
    /**
     * Numeric mode: Prefer integer (int8/int16) paths for embedded to save memory and CPU.
     * When data is quantized (SQ8 or FP16), use DotSq8/L2SqSq8/DotFp16/L2SqFp16 instead of float Dot/L2Sq.
     * Segment scan and runtime already dispatch to these when quant_type is set.
     */
    enum class DistanceMetrics : uint8_t { 
        DOT, 
        L2SQ 
    };

    // ── Canonical Mathematical Vector Kernel ─────────────────────────────────
    // Inner Product: dot(a, b) = sum(a_i * b_i)
    float Dot(std::span<const float> a, std::span<const float> b);
    inline float InnerProduct(std::span<const float> a, std::span<const float> b) {
        return Dot(a, b);
    }

    // Squared Euclidean Distance: sum((a_i - b_i)^2)
    float L2Sq(std::span<const float> a, std::span<const float> b);

    // Euclidean Distance: sqrt(sum((a_i - b_i)^2))
    float L2(std::span<const float> a, std::span<const float> b);

    // Cosine Similarity: dot(a, b) / (||a|| * ||b||) in [-1.0, 1.0]
    // Invariant: returns 0.0f when ||a|| == 0, ||b|| == 0, or dim == 0.
    float CosineSimilarity(std::span<const float> a, std::span<const float> b);

    // Cosine Distance: 1.0f - CosineSimilarity(a, b) in [0.0, 2.0]
    // Returns 1.0f when similarity is 0.0f.
    float CosineDistance(std::span<const float> a, std::span<const float> b);

    /**
     * Computes the canonical ranking score where HIGHER score is ALWAYS better.
     * - kL2: -L2Sq (closer to 0 is better)
     * - kCosine: CosineSimilarity (closer to 1.0 is better)
     * - kInnerProduct: Dot (higher is better)
     */
    float ComputeMetricScore(MetricType metric, std::span<const float> query, std::span<const float> vec);

    // ── Quantized distances (preferred for embedded/edge memory footprint) ──
    // Inner Product for SQ8 quantized codes (int8; preferred for embedded)
    float DotSq8(std::span<const float> query,
                 std::span<const uint8_t> codes,
                 float min_val, float inv_scale, float query_sum = 0.0f);

    /** L2 squared between float query and SQ8 data (min/max dequantize). */
    float L2SqSq8(std::span<const float> query,
                  std::span<const std::uint8_t> data,
                  float min_val, float max_val);

    // Distances for FP16 quantized codes (int16; preferred for embedded)
    float DotFp16(std::span<const float> query, std::span<const uint16_t> codes);
    float L2SqFp16(std::span<const float> query, std::span<const uint16_t> codes);

    // ── Binary Quantization (1-bit; optimized specifically for edge memory reduction) ──
    /** 32x smaller than float vectors. Encodes sign bit. */
    void BitQuantize(std::span<const float> vec, uint8_t* out_codes);
    /** Hamming distance but returning a dot-product-like score (higher=better). */
    float DotBit(std::span<const float> query, std::span<const uint8_t> codes);
    float HammingDist(std::span<const uint8_t> a, std::span<const uint8_t> b);

    // ── Batch distances ──
    void DotBatch(std::span<const float> query,
                  const float* db,
                  std::size_t n,
                  std::uint32_t dim,
                  float* results);

    void L2SqBatch(std::span<const float> query,
                   const float* db,
                  std::size_t n,
                  std::uint32_t dim,
                  float* results);

    /**
     * @brief Vectorized Batch Search (The "Orrify" Pattern).
     * Distilled from DuckDB's vectorized execution.
     */
    void SearchBatch(std::span<const float> query, const FloatBatch& batch, 
                     DistanceMetrics metric, float* results);

    // ── Setup ─────────────────────────────────────────────────────────────────
    void InitDistance();
}
