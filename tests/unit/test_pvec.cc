// test/test_pvec.cc — Comprehensive verification suite for pvec
// Copyright 2026 PomaiDB / pvec authors. MIT License.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "pvec/pvec.h"

#define PVEC_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "ASSERTION FAILED: " << msg << " (" << #cond << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define PVEC_ASSERT_NEAR(a, b, eps, msg) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "ASSERTION FAILED: " << msg << " (" << (a) << " vs " << (b) << ", diff=" << std::abs((a) - (b)) << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

void TestAlignedAlloc() {
    std::cout << "[RUN] TestAlignedAlloc..." << std::endl;
    void* ptr = pvec::aligned_alloc(1024, 64);
    PVEC_ASSERT(ptr != nullptr, "aligned_alloc failed");
    PVEC_ASSERT(reinterpret_cast<uintptr_t>(ptr) % 64 == 0, "pointer not 64-byte aligned");
    pvec::aligned_free(ptr);
    std::cout << "[PASS] TestAlignedAlloc" << std::endl;
}

void TestDistanceMetrics() {
    std::cout << "[RUN] TestDistanceMetrics..." << std::endl;
    const std::size_t dim = 128;
    std::vector<float> a(dim), b(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(i * 2 + 1);
    }

    // Dot product ground truth
    double expected_dot = 0.0;
    for (std::size_t i = 0; i < dim; ++i) expected_dot += a[i] * b[i];
    float dot_val = pvec::dot(a, b);
    PVEC_ASSERT_NEAR(dot_val, expected_dot, 1e-1f, "Dot product mismatch");

    // L2Sq ground truth
    double expected_l2_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        double d = a[i] - b[i];
        expected_l2_sq += d * d;
    }
    float l2_sq_val = pvec::l2_sq(a, b);
    PVEC_ASSERT_NEAR(l2_sq_val, expected_l2_sq, 1e-1f, "L2Sq mismatch");

    // L2 ground truth
    float l2_val = pvec::l2(a, b);
    PVEC_ASSERT_NEAR(l2_val, std::sqrt(expected_l2_sq), 1e-1f, "L2 mismatch");

    // Manhattan ground truth
    double expected_manhattan = 0.0;
    for (std::size_t i = 0; i < dim; ++i) expected_manhattan += std::abs(a[i] - b[i]);
    float manhattan_val = pvec::manhattan(a, b);
    PVEC_ASSERT_NEAR(manhattan_val, expected_manhattan, 1e-1f, "Manhattan mismatch");

    // Cosine similarity
    float cos_sim = pvec::cosine_similarity(a, a);
    PVEC_ASSERT_NEAR(cos_sim, 1.0f, 1e-4f, "Self cosine similarity must be 1.0");

    float cos_dist = pvec::cosine_distance(a, a);
    PVEC_ASSERT_NEAR(cos_dist, 0.0f, 1e-4f, "Self cosine distance must be 0.0");

    std::cout << "[PASS] TestDistanceMetrics" << std::endl;
}

void TestBatchDistances() {
    std::cout << "[RUN] TestBatchDistances..." << std::endl;
    const std::size_t n = 10;
    const std::size_t dim = 64;
    std::vector<float> query(dim, 1.0f);
    std::vector<float> matrix(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) matrix[i] = static_cast<float>(i % 7);

    std::vector<float> dots(n);
    pvec::dot_batch(query, matrix.data(), n, dim, dots.data());
    for (std::size_t i = 0; i < n; ++i) {
        float expected = pvec::dot(query, std::span<const float>(matrix.data() + i * dim, dim));
        PVEC_ASSERT_NEAR(dots[i], expected, 1e-4f, "Batch dot mismatch");
    }

    std::vector<float> l2s(n);
    pvec::l2_sq_batch(query, matrix.data(), n, dim, l2s.data());
    for (std::size_t i = 0; i < n; ++i) {
        float expected = pvec::l2_sq(query, std::span<const float>(matrix.data() + i * dim, dim));
        PVEC_ASSERT_NEAR(l2s[i], expected, 1e-4f, "Batch L2 mismatch");
    }
    std::cout << "[PASS] TestBatchDistances" << std::endl;
}

void TestScalarQuantizer8() {
    std::cout << "[RUN] TestScalarQuantizer8..." << std::endl;
    const std::size_t dim = 128;
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < dim; ++i) vec[i] = -10.0f + static_cast<float>(i) * (20.0f / dim);

    auto params = pvec::ScalarQuantizer8::Calibrate(vec);
    std::vector<uint8_t> codes(dim);
    pvec::ScalarQuantizer8::Encode(vec, codes, params);

    std::vector<float> decoded(dim);
    pvec::ScalarQuantizer8::Decode(codes, decoded, params);

    for (std::size_t i = 0; i < dim; ++i) {
        PVEC_ASSERT_NEAR(decoded[i], vec[i], 0.2f, "SQ8 reconstruction error too high");
    }

    // SIMD DotSq8 vs dequantized dot
    std::vector<float> query(dim, 1.5f);
    float dot_simd = pvec::ScalarQuantizer8::Dot(query, codes, params.min_val, params.inv_scale);
    float dot_expected = pvec::dot(query, decoded);
    PVEC_ASSERT_NEAR(dot_simd, dot_expected, 0.5f, "DotSq8 mismatch against dequantized dot");

    // SIMD L2SqSq8 vs dequantized L2Sq
    float l2_simd = pvec::ScalarQuantizer8::L2Sq(query, codes, params.min_val, params.max_val);
    float l2_expected = pvec::l2_sq(query, decoded);
    PVEC_ASSERT_NEAR(l2_simd, l2_expected, 1.0f, "L2SqSq8 mismatch against dequantized L2Sq");

    std::cout << "[PASS] TestScalarQuantizer8" << std::endl;
}

void TestScalarQuantizer4() {
    std::cout << "[RUN] TestScalarQuantizer4..." << std::endl;
    const std::size_t dim = 64;
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < dim; ++i) vec[i] = static_cast<float>(i);

    std::vector<uint8_t> codes((dim + 1) / 2);
    pvec::ScalarQuantizer4::Encode(vec, codes, 0.0f, static_cast<float>(dim - 1));

    std::vector<float> decoded(dim);
    pvec::ScalarQuantizer4::Decode(codes, decoded, 0.0f, static_cast<float>(dim - 1));

    for (std::size_t i = 0; i < dim; ++i) {
        PVEC_ASSERT_NEAR(decoded[i], vec[i], 3.0f, "SQ4 reconstruction error too high");
    }
    std::cout << "[PASS] TestScalarQuantizer4" << std::endl;
}

void TestHalfPrecision() {
    std::cout << "[RUN] TestHalfPrecision..." << std::endl;
    float test_vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 3.14159f, 65504.0f, 0.000061f};
    for (float v : test_vals) {
        uint16_t h = pvec::float_to_fp16(v);
        float back = pvec::fp16_to_float(h);
        PVEC_ASSERT_NEAR(back, v, std::abs(v) * 0.01f + 1e-4f, "FP16 roundtrip failed");
    }

    const std::size_t dim = 64;
    std::vector<float> vec(dim);
    for (std::size_t i = 0; i < dim; ++i) vec[i] = static_cast<float>(i) * 0.5f;

    std::vector<uint16_t> fp16_codes(dim);
    pvec::HalfFloatQuantizer::Encode(vec, fp16_codes);

    float dot_val = pvec::HalfFloatQuantizer::Dot(vec, fp16_codes);
    float dot_exact = pvec::dot(vec, vec);
    PVEC_ASSERT_NEAR(dot_val, dot_exact, dot_exact * 0.005f, "HalfFloatQuantizer::Dot error too high");
    std::cout << "[PASS] TestHalfPrecision" << std::endl;
}

void TestBitQuantizer() {
    std::cout << "[RUN] TestBitQuantizer..." << std::endl;
    const std::size_t dim = 128;
    std::vector<float> a(dim), b(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        a[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        b[i] = (i % 3 == 0) ? 1.0f : -1.0f;
    }

    std::vector<uint8_t> codes_a(dim / 8), codes_b(dim / 8);
    pvec::BitQuantizer::Encode(a, codes_a.data());
    pvec::BitQuantizer::Encode(b, codes_b.data());

    uint32_t ham_dist = pvec::BitQuantizer::HammingDistance(codes_a, codes_b);
    PVEC_ASSERT(ham_dist > 0 && ham_dist < dim, "Hamming distance must be in (0, dim)");

    uint32_t self_dist = pvec::BitQuantizer::HammingDistance(codes_a, codes_a);
    PVEC_ASSERT(self_dist == 0, "Self Hamming distance must be 0");

    float score = pvec::BitQuantizer::DotBit(codes_a, codes_a, dim);
    PVEC_ASSERT_NEAR(score, static_cast<float>(dim), 1e-4f, "Self DotBit score must be dim");
    std::cout << "[PASS] TestBitQuantizer" << std::endl;
}

void TestKMeans() {
    std::cout << "[RUN] TestKMeans..." << std::endl;
    const std::size_t n = 200;
    const std::size_t dim = 4;
    const std::size_t k = 3;

    std::vector<float> data(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        float base = static_cast<float>((i % 3) * 100);
        for (std::size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = base + static_cast<float>(i % 5);
        }
    }

    auto centroids = pvec::KMeans::Fit(data.data(), n, dim, k);
    PVEC_ASSERT(centroids.size() == k * dim, "Centroids size mismatch");
    std::cout << "[PASS] TestKMeans" << std::endl;
}

void TestProductQuantizer() {
    std::cout << "[RUN] TestProductQuantizer..." << std::endl;
    const std::size_t n = 256;
    const std::size_t dim = 16;
    const std::size_t m = 4;

    std::vector<float> data(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) {
        data[i] = static_cast<float>((i * 17) % 100) * 0.1f;
    }

    pvec::ProductQuantizer pq;
    pvec::PQConfig cfg;
    cfg.num_subspaces = m;
    cfg.centroids_per_subspace = 16;
    cfg.kmeans_iterations = 5;

    bool ok = pq.Train(data.data(), n, dim, cfg);
    PVEC_ASSERT(ok, "PQ training failed");
    PVEC_ASSERT(pq.is_trained(), "PQ not trained");

    std::vector<uint8_t> codes(m);
    pq.Encode(std::span<const float>(data.data(), dim), codes.data());

    std::vector<float> lut(m * cfg.centroids_per_subspace);
    pq.ComputeL2DistanceTable(std::span<const float>(data.data(), dim), lut.data());

    float dist = pq.ComputeDistanceWithTable(lut.data(), codes.data());
    PVEC_ASSERT(dist >= 0.0f, "ADC distance must be non-negative");
    std::cout << "[PASS] TestProductQuantizer" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  pvec Verification Suite v" << pvec::kVersion << std::endl;
    std::cout << "========================================" << std::endl;

    TestAlignedAlloc();
    TestDistanceMetrics();
    TestBatchDistances();
    TestScalarQuantizer8();
    TestScalarQuantizer4();
    TestHalfPrecision();
    TestBitQuantizer();
    TestKMeans();
    TestProductQuantizer();

    std::cout << "========================================" << std::endl;
    std::cout << "  ALL 9 SUITES PASSED CLEANLY (100%)" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
