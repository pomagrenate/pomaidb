// tests/unit/core_compute_test.cc — Verification of PomaiDB Custom Core Compute Accelerator
// Copyright 2026 PomaiDB authors. MIT License.

#include "pomegranate_compute.h"
#include "distance.h"
#include "pulp.h"
#include "seed_scar.h"
#include "compass.h"
#include "locule.h"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>
#include <filesystem>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " -> " << #cond << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        float diff = std::fabs((a) - (b)); \
        if (diff > (eps)) { \
            std::cerr << "Near assertion failed at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << #a << " (" << (a) << ") vs " << #b << " (" << (b) \
                      << "), diff=" << diff << " > " << (eps) << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void TestDotSq8_4x_Equivalence() {
    std::cout << "[TEST] TestDotSq8_4x_Equivalence..." << std::flush;
    constexpr size_t dim = 128;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> q_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> c_dist(0, 255);

    std::vector<float> query(dim);
    float q_sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        query[i] = q_dist(rng);
        q_sum += query[i];
    }

    std::vector<uint8_t> c0(dim), c1(dim), c2(dim), c3(dim);
    for (size_t i = 0; i < dim; ++i) {
        c0[i] = static_cast<uint8_t>(c_dist(rng));
        c1[i] = static_cast<uint8_t>(c_dist(rng));
        c2[i] = static_cast<uint8_t>(c_dist(rng));
        c3[i] = static_cast<uint8_t>(c_dist(rng));
    }

    float min_val = -2.5f;
    float inv_scale = 0.0196f; // 5.0f / 255.0f

    float ref[4];
    ref[0] = pomai::core::DotSq8(query, c0, min_val, inv_scale, q_sum);
    ref[1] = pomai::core::DotSq8(query, c1, min_val, inv_scale, q_sum);
    ref[2] = pomai::core::DotSq8(query, c2, min_val, inv_scale, q_sum);
    ref[3] = pomai::core::DotSq8(query, c3, min_val, inv_scale, q_sum);

    float sim_scores[4];
    pomai::compute::DotSq8_4x(query.data(), c0.data(), c1.data(), c2.data(), c3.data(),
                              dim, min_val, inv_scale, q_sum, sim_scores);

    for (int k = 0; k < 4; ++k) {
        TEST_ASSERT_NEAR(ref[k], sim_scores[k], 1e-3f);
    }
    std::cout << " PASSED" << std::endl;
}

void TestL2SqSq8_4x_Equivalence() {
    std::cout << "[TEST] TestL2SqSq8_4x_Equivalence..." << std::flush;
    constexpr size_t dim = 128;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> q_dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> c_dist(0, 255);

    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = q_dist(rng);
    }

    std::vector<uint8_t> c0(dim), c1(dim), c2(dim), c3(dim);
    for (size_t i = 0; i < dim; ++i) {
        c0[i] = static_cast<uint8_t>(c_dist(rng));
        c1[i] = static_cast<uint8_t>(c_dist(rng));
        c2[i] = static_cast<uint8_t>(c_dist(rng));
        c3[i] = static_cast<uint8_t>(c_dist(rng));
    }

    float min_val = -3.0f;
    float max_val = 3.0f;

    float ref[4];
    ref[0] = pomai::core::L2SqSq8(query, c0, min_val, max_val);
    ref[1] = pomai::core::L2SqSq8(query, c1, min_val, max_val);
    ref[2] = pomai::core::L2SqSq8(query, c2, min_val, max_val);
    ref[3] = pomai::core::L2SqSq8(query, c3, min_val, max_val);

    float sim_dists[4];
    pomai::compute::L2SqSq8_4x(query.data(), c0.data(), c1.data(), c2.data(), c3.data(),
                               dim, min_val, max_val, sim_dists);

    for (int k = 0; k < 4; ++k) {
        TEST_ASSERT_NEAR(ref[k], sim_dists[k], 1e-3f);
    }
    std::cout << " PASSED" << std::endl;
}

void TestDotF32_4x_Equivalence() {
    std::cout << "[TEST] TestDotF32_4x_Equivalence..." << std::flush;
    constexpr size_t dim = 256;
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    for (size_t i = 0; i < dim; ++i) {
        q[i] = dist(rng);
        v0[i] = dist(rng);
        v1[i] = dist(rng);
        v2[i] = dist(rng);
        v3[i] = dist(rng);
    }

    float ref[4];
    ref[0] = pomai::core::Dot(q, v0);
    ref[1] = pomai::core::Dot(q, v1);
    ref[2] = pomai::core::Dot(q, v2);
    ref[3] = pomai::core::Dot(q, v3);

    float out[4];
    pomai::compute::DotF32_4x(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, out);

    for (int k = 0; k < 4; ++k) {
        TEST_ASSERT_NEAR(ref[k], out[k], 1e-3f);
    }
    std::cout << " PASSED" << std::endl;
}

void TestL2SqF32_4x_Equivalence() {
    std::cout << "[TEST] TestL2SqF32_4x_Equivalence..." << std::flush;
    constexpr size_t dim = 256;
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    for (size_t i = 0; i < dim; ++i) {
        q[i] = dist(rng);
        v0[i] = dist(rng);
        v1[i] = dist(rng);
        v2[i] = dist(rng);
        v3[i] = dist(rng);
    }

    float ref[4];
    ref[0] = pomai::core::L2Sq(q, v0);
    ref[1] = pomai::core::L2Sq(q, v1);
    ref[2] = pomai::core::L2Sq(q, v2);
    ref[3] = pomai::core::L2Sq(q, v3);

    float out[4];
    pomai::compute::L2SqF32_4x(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, out);

    for (int k = 0; k < 4; ++k) {
        TEST_ASSERT_NEAR(ref[k], out[k], 1e-3f);
    }
    std::cout << " PASSED" << std::endl;
}

void TestFastBoundedTopKHeap() {
    std::cout << "[TEST] TestFastBoundedTopKHeap..." << std::flush;
    pomai::compute::FastBoundedTopKHeap heap(5);
    TEST_ASSERT(heap.size() == 0);
    TEST_ASSERT(!heap.full());

    heap.Push({1, 10.0f, nullptr, 0, false});
    heap.Push({2, 50.0f, nullptr, 0, false});
    heap.Push({3, 30.0f, nullptr, 0, false});
    heap.Push({4, 20.0f, nullptr, 0, false});
    heap.Push({5, 40.0f, nullptr, 0, false});

    TEST_ASSERT(heap.full());
    TEST_ASSERT(heap.size() == 5);
    TEST_ASSERT(heap.worst_score() == 10.0f);

    // Rejection of score <= worst_score
    bool pushed = heap.Push({6, 5.0f, nullptr, 0, false});
    TEST_ASSERT(!pushed);
    TEST_ASSERT(heap.size() == 5);
    TEST_ASSERT(heap.worst_score() == 10.0f);

    // Replacement of worst
    pushed = heap.Push({7, 25.0f, nullptr, 0, false});
    TEST_ASSERT(pushed);
    TEST_ASSERT(heap.size() == 5);
    TEST_ASSERT(heap.worst_score() == 20.0f);

    auto sorted = heap.ExtractSortedDesc();
    TEST_ASSERT(sorted.size() == 5);
    // Expected sorted order: 50, 40, 30, 25, 20
    TEST_ASSERT(sorted[0].score == 50.0f);
    TEST_ASSERT(sorted[1].score == 40.0f);
    TEST_ASSERT(sorted[2].score == 30.0f);
    TEST_ASSERT(sorted[3].score == 25.0f);
    TEST_ASSERT(sorted[4].score == 20.0f);

    std::cout << " PASSED" << std::endl;
}

void TestPulpBatchScanner() {
    std::cout << "[TEST] TestPulpBatchScanner..." << std::flush;
    constexpr uint32_t dim = 64;
    pomai::storage::PulpBuilder builder(dim, 1);

    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> raw_vecs(8, std::vector<float>(dim));
    std::vector<std::span<const float>> spans;
    for (int i = 0; i < 8; ++i) {
        for (uint32_t d = 0; d < dim; ++d) {
            raw_vecs[i][d] = dist(rng);
        }
        spans.push_back(raw_vecs[i]);
    }
    builder.Train(spans);
    for (int i = 0; i < 8; ++i) {
        builder.EncodeAppend(spans[i]);
    }

    pomai::storage::PulpView pulp(builder.buffer().data(), builder.count(), builder.dim(),
                                 builder.min_val(), builder.inv_scale(), builder.quant_type());

    std::vector<float> query(dim);
    float q_sum = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        query[d] = dist(rng);
        q_sum += query[d];
    }

    // Test Scan4 for InnerProduct
    float scores_out[4];
    pomai::compute::PulpBatchScanner::Scan4(query.data(), dim, pomai::MetricType::kInnerProduct,
                                            q_sum, pulp, 0, 4, scores_out);

    for (uint32_t i = 0; i < 4; ++i) {
        float ref = pulp.Taste(query, i, pomai::MetricType::kInnerProduct, q_sum);
        TEST_ASSERT_NEAR(ref, scores_out[i], 1e-3f);
    }

    // Test Scan4 for L2
    pomai::compute::PulpBatchScanner::Scan4(query.data(), dim, pomai::MetricType::kL2,
                                            q_sum, pulp, 4, 4, scores_out);

    for (uint32_t i = 0; i < 4; ++i) {
        float ref = pulp.Taste(query, 4 + i, pomai::MetricType::kL2, q_sum);
        TEST_ASSERT_NEAR(ref, scores_out[i], 1e-3f);
    }

    std::cout << " PASSED" << std::endl;
}

void TestSeedScar_IsDeleted4() {
    std::cout << "[TEST] TestSeedScar_IsDeleted4..." << std::flush;
    constexpr uint32_t count = 70;
    pomai::storage::SeedScarBuilder builder(count);

    std::vector<uint32_t> deleted_slots = {0, 3, 5, 7, 8, 9, 14, 15, 16, 23, 24, 31, 32, 63, 67, 68, 69};
    for (uint32_t slot : deleted_slots) {
        builder.MarkDeleted(slot);
    }

    pomai::storage::SeedScarView view(builder.bytes().data(), count);

    for (uint32_t s = 0; s < count; ++s) {
        uint8_t mask = view.IsDeleted4(s);
        for (uint32_t k = 0; k < 4; ++k) {
            bool bit_expected = (s + k < count) ? view.IsDeleted(s + k) : false;
            bool bit_actual = (mask & (1u << k)) != 0;
            TEST_ASSERT(bit_expected == bit_actual);
        }
    }
    std::cout << " PASSED" << std::endl;
}

void TestPulpBuilder_Avx2_Quantization() {
    std::cout << "[TEST] TestPulpBuilder_Avx2_Quantization..." << std::flush;
    constexpr uint32_t dim = 128;
    constexpr size_t num_vecs = 32;
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<std::vector<float>> vecs(num_vecs, std::vector<float>(dim));
    std::vector<std::span<const float>> spans;
    spans.reserve(num_vecs);

    for (size_t i = 0; i < num_vecs; ++i) {
        for (uint32_t d = 0; d < dim; ++d) {
            vecs[i][d] = dist(rng);
        }
        spans.push_back(vecs[i]);
    }

    pomai::storage::PulpBuilder builder(dim);
    builder.Train(spans);

    TEST_ASSERT(builder.min_val() < builder.min_val() + 255.0f * builder.inv_scale());

    for (size_t i = 0; i < num_vecs; ++i) {
        builder.EncodeAppend(spans[i]);
    }

    TEST_ASSERT(builder.count() == num_vecs);
    TEST_ASSERT(builder.buffer().size() == num_vecs * dim);

    float min_val = builder.min_val();
    float scale = (builder.inv_scale() > 0.0f) ? (1.0f / builder.inv_scale()) : 0.0f;

    for (size_t i = 0; i < num_vecs; ++i) {
        const uint8_t* encoded = builder.buffer().data() + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            float normalized = (vecs[i][d] - min_val) * scale;
            float clamped = std::clamp(normalized, 0.0f, 255.0f);
            uint8_t expected = static_cast<uint8_t>(std::round(clamped));
            int diff = std::abs(static_cast<int>(encoded[d]) - static_cast<int>(expected));
            TEST_ASSERT(diff <= 1);
        }
    }
    std::cout << " PASSED" << std::endl;
}

void TestCompass_Orient4_Equivalence() {
    std::cout << "[TEST] TestCompass_Orient4_Equivalence..." << std::flush;
    constexpr uint32_t dim = 128;
    constexpr size_t num_locules = 9;
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    pomai::routing::Compass compass_l2(pomai::MetricType::kL2);
    pomai::routing::Compass compass_ip(pomai::MetricType::kInnerProduct);

    std::string test_dir = "./test_compass_core_tmp";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directories(test_dir);

    std::vector<pomai::alloc::SharedPtr<pomai::storage::Locule>> locules;
    for (size_t l = 0; l < num_locules; ++l) {
        std::string loc_path = test_dir + "/locule_" + std::to_string(l) + ".pom";
        pomai::format::LoculeAnchor anchor;
        anchor.id = static_cast<uint32_t>(l + 1);
        anchor.radius = 1.5f + static_cast<float>(l) * 0.2f;
        anchor.centroid.resize(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            anchor.centroid[d] = dist(rng);
        }
        (void)pomai::storage::Locule::Write(loc_path, anchor.id, 1, dim, anchor, {});
        pomai::alloc::SharedPtr<pomai::storage::Locule> loc_ptr;
        auto st = pomai::storage::Locule::Open(loc_path, &loc_ptr);
        TEST_ASSERT(st.ok() && loc_ptr != nullptr);
        locules.push_back(loc_ptr);
    }

    compass_l2.UpdateLocules(locules);
    compass_ip.UpdateLocules(locules);

    TEST_ASSERT(compass_l2.matrix().count == num_locules);
    TEST_ASSERT(compass_l2.matrix().dim == dim);

    std::vector<float> query(dim);
    for (uint32_t d = 0; d < dim; ++d) query[d] = dist(rng);

    auto oriented_l2 = compass_l2.Orient(query, 5);
    TEST_ASSERT(oriented_l2.size() == 5);
    for (size_t i = 1; i < oriented_l2.size(); ++i) {
        TEST_ASSERT(oriented_l2[i - 1].distance_to_centroid <= oriented_l2[i].distance_to_centroid);
    }

    auto oriented_ip = compass_ip.Orient(query, 5);
    TEST_ASSERT(oriented_ip.size() == 5);
    for (size_t i = 1; i < oriented_ip.size(); ++i) {
        TEST_ASSERT(oriented_ip[i - 1].distance_to_centroid >= oriented_ip[i].distance_to_centroid);
    }

    compass_l2.UpdateLocules({});
    compass_ip.UpdateLocules({});
    for (auto& loc : locules) {
        if (loc) loc->CloseMapping();
    }
    locules.clear();

    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
    std::cout << " PASSED" << std::endl;
}

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "   POMAIDB CUSTOM CORE COMPUTE ACCELERATOR TESTS     " << std::endl;
    std::cout << "======================================================" << std::endl;

    pomai::core::InitDistance();

    auto t0 = std::chrono::high_resolution_clock::now();

    TestDotSq8_4x_Equivalence();
    TestL2SqSq8_4x_Equivalence();
    TestDotF32_4x_Equivalence();
    TestL2SqF32_4x_Equivalence();
    TestFastBoundedTopKHeap();
    TestPulpBatchScanner();
    TestSeedScar_IsDeleted4();
    TestPulpBuilder_Avx2_Quantization();
    TestCompass_Orient4_Equivalence();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "======================================================" << std::endl;
    std::cout << " ALL 9 CORE COMPUTE SUITES PASSED CLEANLY in " << elapsed << "s" << std::endl;
    std::cout << "======================================================" << std::endl;
    return 0;
}
