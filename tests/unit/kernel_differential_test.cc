// kernel_differential_test.cc — Differential testing and property test suite
//
// Compares optimized SIMD kernels against the pure scalar reference oracle
// across arbitrary dimensions, unaligned memory, numeric extremes, and quantization.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "tests/common/test_main.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "distance.h"
#include "reference_distance.h"
#include "topk.h"
#include "pulp.h"
#include "scalar_quantizer.h"
#include "bit_quantizer.h"
#include "utils/half_float.h"

namespace pomai {
namespace {

std::vector<float> MakeUniform(size_t n, float low, float high, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

bool ApproxEq(double a, double b, double eps = 1e-4) {
    if (std::isnan(a) || std::isnan(b)) return false;
    double diff = std::abs(a - b);
    if (diff <= eps) return true;
    double max_mag = std::max(std::abs(a), std::abs(b));
    return (diff / max_mag) <= eps;
}

// 1. Dimensions Test Matrix: All SIMD tail and arbitrary boundary dimensions
POMAI_TEST(Kernel_Differential_AllDimensions_SIMD_vs_Reference) {
    core::InitDistance();

    const std::vector<size_t> dims = {
        0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        127, 128, 129, 255, 256, 257, 1024, 4096
    };

    for (size_t d : dims) {
        if (d == 0) {
            std::span<const float> empty;
            POMAI_EXPECT_EQ(core::Dot(empty, empty), 0.0f);
            POMAI_EXPECT_EQ(core::L2Sq(empty, empty), 0.0f);
            POMAI_EXPECT_EQ(core::L2(empty, empty), 0.0f);
            POMAI_EXPECT_EQ(core::CosineSimilarity(empty, empty), 0.0f);
            POMAI_EXPECT_EQ(core::CosineDistance(empty, empty), 1.0f);
            continue;
        }

        auto a = MakeUniform(d, -1.0f, 1.0f, static_cast<uint32_t>(d * 17 + 1));
        auto b = MakeUniform(d, -1.0f, 1.0f, static_cast<uint32_t>(d * 31 + 7));

        std::span<const float> sa(a);
        std::span<const float> sb(b);

        // Dot / InnerProduct
        float dot_simd = core::Dot(sa, sb);
        double dot_ref = core::reference::InnerProduct(sa, sb);
        POMAI_EXPECT_TRUE(ApproxEq(dot_simd, dot_ref, 2e-4));

        // L2Sq
        float l2sq_simd = core::L2Sq(sa, sb);
        double l2sq_ref = core::reference::L2Sq(sa, sb);
        POMAI_EXPECT_TRUE(ApproxEq(l2sq_simd, l2sq_ref, 2e-4));

        // L2
        float l2_simd = core::L2(sa, sb);
        double l2_ref = core::reference::L2(sa, sb);
        POMAI_EXPECT_TRUE(ApproxEq(l2_simd, l2_ref, 2e-4));

        // Cosine Similarity
        float cos_simd = core::CosineSimilarity(sa, sb);
        double cos_ref = core::reference::CosineSimilarity(sa, sb);
        POMAI_EXPECT_TRUE(ApproxEq(cos_simd, cos_ref, 1e-4));

        // Cosine Distance
        float cos_dist_simd = core::CosineDistance(sa, sb);
        double cos_dist_ref = core::reference::CosineDistance(sa, sb);
        POMAI_EXPECT_TRUE(ApproxEq(cos_dist_simd, cos_dist_ref, 1e-4));
    }
}

// 2. Unaligned Pointer and Tail Safety Test
POMAI_TEST(Kernel_Differential_UnalignedMemorySafety) {
    core::InitDistance();

    const size_t dim = 129; // Not a multiple of 8 or 16
    const size_t pad = 8;
    std::vector<float> buf_a(dim + pad);
    std::vector<float> buf_b(dim + pad);

    for (size_t i = 0; i < dim + pad; ++i) {
        buf_a[i] = static_cast<float>(i % 11) * 0.1f - 0.5f;
        buf_b[i] = static_cast<float>(i % 13) * 0.1f - 0.5f;
    }

    // Test different unaligned offset pairings (1, 2, 3 bytes / float elements)
    for (size_t offset_a = 0; offset_a < 4; ++offset_a) {
        for (size_t offset_b = 0; offset_b < 4; ++offset_b) {
            std::span<const float> sa(buf_a.data() + offset_a, dim);
            std::span<const float> sb(buf_b.data() + offset_b, dim);

            float dot_simd = core::Dot(sa, sb);
            double dot_ref = core::reference::InnerProduct(sa, sb);
            POMAI_EXPECT_TRUE(ApproxEq(dot_simd, dot_ref, 2e-4));

            float l2sq_simd = core::L2Sq(sa, sb);
            double l2sq_ref = core::reference::L2Sq(sa, sb);
            POMAI_EXPECT_TRUE(ApproxEq(l2sq_simd, l2sq_ref, 2e-4));

            float cos_simd = core::CosineSimilarity(sa, sb);
            double cos_ref = core::reference::CosineSimilarity(sa, sb);
            POMAI_EXPECT_TRUE(ApproxEq(cos_simd, cos_ref, 1e-4));
        }
    }
}

// 3. Mathematical Properties (Symmetry, Identity, Scaling Invariance)
POMAI_TEST(Kernel_Differential_MathematicalProperties) {
    core::InitDistance();

    const size_t dim = 64;
    auto q = MakeUniform(dim, -2.0f, 2.0f, 42);
    auto v = MakeUniform(dim, -2.0f, 2.0f, 1337);

    std::span<const float> sq(q);
    std::span<const float> sv(v);

    // Identity: L2(a, a) == 0, Cosine(a, a) == 1.0
    POMAI_EXPECT_TRUE(ApproxEq(core::L2Sq(sq, sq), 0.0, 1e-6));
    POMAI_EXPECT_TRUE(ApproxEq(core::CosineSimilarity(sq, sq), 1.0, 1e-5));
    POMAI_EXPECT_TRUE(ApproxEq(core::CosineDistance(sq, sq), 0.0, 1e-5));

    // Symmetry: L2(a, b) == L2(b, a), Dot(a, b) == Dot(b, a), Cosine(a, b) == Cosine(b, a)
    POMAI_EXPECT_TRUE(ApproxEq(core::L2Sq(sq, sv), core::L2Sq(sv, sq), 1e-5));
    POMAI_EXPECT_TRUE(ApproxEq(core::Dot(sq, sv), core::Dot(sv, sq), 1e-5));
    POMAI_EXPECT_TRUE(ApproxEq(core::CosineSimilarity(sq, sv), core::CosineSimilarity(sv, sq), 1e-5));

    // Scaling Invariance of Cosine: for alpha > 0, cos(q, alpha*v) == cos(q, v)
    for (float alpha : {0.001f, 0.5f, 2.0f, 10.0f, 1000.0f}) {
        std::vector<float> v_scaled(dim);
        for (size_t i = 0; i < dim; ++i) v_scaled[i] = v[i] * alpha;
        std::span<const float> sv_scaled(v_scaled);

        float sim_orig = core::CosineSimilarity(sq, sv);
        float sim_scaled = core::CosineSimilarity(sq, sv_scaled);
        POMAI_EXPECT_TRUE(ApproxEq(sim_orig, sim_scaled, 1e-4));
    }
}

// 4. Zero Vector and Extreme Float Contract
POMAI_TEST(Kernel_Differential_ZeroVectorAndExtremes) {
    core::InitDistance();

    const size_t dim = 32;
    std::vector<float> zero_vec(dim, 0.0f);
    auto normal_vec = MakeUniform(dim, 1.0f, 5.0f, 123);

    std::span<const float> sz(zero_vec);
    std::span<const float> sn(normal_vec);

    // Contract: Zero vector cosine similarity is explicitly 0.0f, distance is 1.0f
    POMAI_EXPECT_EQ(core::CosineSimilarity(sz, sn), 0.0f);
    POMAI_EXPECT_EQ(core::CosineSimilarity(sn, sz), 0.0f);
    POMAI_EXPECT_EQ(core::CosineSimilarity(sz, sz), 0.0f);
    POMAI_EXPECT_EQ(core::CosineDistance(sz, sn), 1.0f);
    POMAI_EXPECT_EQ(core::CosineDistance(sn, sz), 1.0f);
    POMAI_EXPECT_EQ(core::CosineDistance(sz, sz), 1.0f);

    // Extreme values: very small magnitude
    std::vector<float> tiny_a(dim, 1e-15f);
    std::vector<float> tiny_b(dim, 2e-15f);
    float dot_tiny = core::Dot(tiny_a, tiny_b);
    POMAI_EXPECT_TRUE(std::isfinite(dot_tiny));

    // All negative values
    auto neg_a = MakeUniform(dim, -100.0f, -1.0f, 77);
    auto neg_b = MakeUniform(dim, -50.0f, -0.5f, 88);
    POMAI_EXPECT_TRUE(ApproxEq(core::Dot(neg_a, neg_b), core::reference::InnerProduct(neg_a, neg_b), 1e-3));
}

// 5. Top-K Determinism & Tie-Breaking
POMAI_TEST(Kernel_Differential_TopKDeterminism) {
    // 5 candidates with identical score = 0.85f, but varying IDs
    std::vector<core::TopKItem> items = {
        {100, 0.85f, 0, nullptr},
        {5,   0.85f, 0, nullptr},
        {42,  0.85f, 0, nullptr},
        {1,   0.85f, 0, nullptr},
        {999, 0.85f, 0, nullptr},
        {7,   0.95f, 0, nullptr}, // higher score
        {2,   0.10f, 0, nullptr}  // lower score
    };

    core::SelectTopK(items, 4);

    POMAI_EXPECT_EQ(items.size(), 4u);
    // Rank 1: score 0.95 (ID 7)
    POMAI_EXPECT_EQ(items[0].id, 7u);
    POMAI_EXPECT_EQ(items[0].score, 0.95f);

    // Rank 2, 3, 4: all have score 0.85, so must be ordered by ID ascending: 1, 5, 42
    POMAI_EXPECT_EQ(items[1].id, 1u);
    POMAI_EXPECT_EQ(items[1].score, 0.85f);
    POMAI_EXPECT_EQ(items[2].id, 5u);
    POMAI_EXPECT_EQ(items[2].score, 0.85f);
    POMAI_EXPECT_EQ(items[3].id, 42u);
    POMAI_EXPECT_EQ(items[3].score, 0.85f);

    // Test BoundedTopKQueue equivalence
    core::BoundedTopKQueue queue(4);
    std::vector<core::TopKItem> raw_items = {
        {100, 0.85f, 0, nullptr},
        {5,   0.85f, 0, nullptr},
        {42,  0.85f, 0, nullptr},
        {1,   0.85f, 0, nullptr},
        {999, 0.85f, 0, nullptr},
        {7,   0.95f, 0, nullptr},
        {2,   0.10f, 0, nullptr}
    };
    for (const auto& it : raw_items) {
        queue.Push(it);
    }
    auto q_sorted = queue.ExtractSorted();
    POMAI_EXPECT_EQ(q_sorted.size(), 4u);
    for (size_t i = 0; i < 4; ++i) {
        POMAI_EXPECT_EQ(q_sorted[i].id, items[i].id);
        POMAI_EXPECT_EQ(q_sorted[i].score, items[i].score);
    }
}

// 6. Quantization Boundaries and Robustness
POMAI_TEST(Kernel_Differential_QuantizationBoundaries) {
    const size_t dim = 16;

    // SQ8 with constant vector (0-range)
    core::ScalarQuantizer8Bit sq8(dim);
    std::vector<float> const_data(dim * 4, 3.14f);
    POMAI_EXPECT_OK(sq8.Train(const_data, 4));

    auto codes = sq8.Encode(std::span<const float>(const_data.data(), dim));
    POMAI_EXPECT_EQ(codes.size(), dim);
    for (uint8_t c : codes) {
        POMAI_EXPECT_EQ(c, 0u);
    }

    auto decoded = sq8.Decode(codes);
    POMAI_EXPECT_EQ(decoded.size(), dim);
    for (float f : decoded) {
        POMAI_EXPECT_TRUE(ApproxEq(f, 3.14f, 1e-4));
    }

    // PulpBuilder with empty vectors (prevents float overflow to Inf and NaN)
    storage::PulpBuilder pb(dim);
    std::vector<std::span<const float>> empty_vecs;
    pb.Train(empty_vecs);
    POMAI_EXPECT_TRUE(std::isfinite(pb.min_val()));
    POMAI_EXPECT_TRUE(std::isfinite(pb.inv_scale()));

    // FP16 roundtrip
    float original_f = 12.345f;
    uint16_t h = util::float32_to_float16(original_f);
    float roundtrip_f = util::float16_to_float32(h);
    POMAI_EXPECT_TRUE(ApproxEq(original_f, roundtrip_f, 1e-3));

    // BitQuantizer with mixed signs
    core::BitQuantizer bq(dim);
    std::vector<float> mixed = {1.0f, -1.0f, 2.5f, -0.5f, 0.0f, 4.0f, -3.0f, 0.1f,
                                -0.2f, 0.9f, -1.5f, 2.0f, -0.1f, 0.8f, -2.0f, 1.2f};
    auto bit_codes = bq.Encode(mixed);
    POMAI_EXPECT_EQ(bit_codes.size(), (dim + 7) / 8);
    auto bit_decoded = bq.Decode(bit_codes);
    POMAI_EXPECT_EQ(bit_decoded.size(), dim);
    for (size_t i = 0; i < dim; ++i) {
        if (mixed[i] > 0.0f) {
            POMAI_EXPECT_EQ(bit_decoded[i], 1.0f);
        } else {
            POMAI_EXPECT_EQ(bit_decoded[i], -1.0f);
        }
    }
}

} // namespace
} // namespace pomai
