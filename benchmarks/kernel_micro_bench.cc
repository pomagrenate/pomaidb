// kernel_micro_bench.cc — Dedicated Kernel Microbenchmarks
//
// Benchmarks Scalar Reference vs Optimized SIMD for:
// - L2Sq
// - Inner Product (Dot)
// - Cosine Similarity
// - SQ8 Quantized Distance
// - FP16 Distance
// - Top-K Selection (Quickselect vs Full Sort)
// Across dimensions: 32, 64, 128, 256, 512, 768, 1024, 1536, 2048.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "distance.h"
#include "reference_distance.h"
#include "topk.h"
#include "utils/half_float.h"

namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> MakeRandomVec(size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = dist(rng);
    return v;
}

struct BenchResult {
    double scalar_ns = 0.0;
    double simd_ns = 0.0;
    double speedup = 0.0;
    double simd_mvec_s = 0.0;
    double simd_gb_s = 0.0;
};

template <typename ScalarFn, typename SimdFn>
BenchResult RunBench(size_t dim, size_t iterations, ScalarFn scalar_fn, SimdFn simd_fn) {
    auto a = MakeRandomVec(dim, 101);
    auto b = MakeRandomVec(dim, 202);

    // Warmup
    volatile float dummy = 0.0f;
    for (size_t i = 0; i < 1000; ++i) {
        dummy += static_cast<float>(scalar_fn(a, b));
        dummy += simd_fn(a, b);
    }

    // Benchmark Scalar
    const auto t0_scalar = Clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        dummy += static_cast<float>(scalar_fn(a, b));
    }
    const auto t1_scalar = Clock::now();

    // Benchmark SIMD
    const auto t0_simd = Clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        dummy += simd_fn(a, b);
    }
    const auto t1_simd = Clock::now();

    double sec_scalar = std::chrono::duration<double>(t1_scalar - t0_scalar).count();
    double sec_simd = std::chrono::duration<double>(t1_simd - t0_simd).count();

    BenchResult r;
    r.scalar_ns = (sec_scalar / static_cast<double>(iterations)) * 1e9;
    r.simd_ns = (sec_simd / static_cast<double>(iterations)) * 1e9;
    r.speedup = r.simd_ns > 0.0 ? (r.scalar_ns / r.simd_ns) : 1.0;
    r.simd_mvec_s = r.simd_ns > 0.0 ? (1e3 / r.simd_ns) : 0.0;
    // 2 vectors * dim * 4 bytes
    double bytes_per_op = static_cast<double>(dim * 2 * sizeof(float));
    r.simd_gb_s = sec_simd > 0.0 ? ((bytes_per_op * static_cast<double>(iterations)) / (sec_simd * 1e9)) : 0.0;

    // Prevent compiler optimizing dummy away
    if (dummy == 12345.67f) std::cout << dummy;
    return r;
}

} // namespace

int main() {
    pomai::core::InitDistance();

    const std::vector<size_t> dims = {32, 64, 128, 256, 512, 768, 1024, 1536, 2048};
    const size_t iters = 200000;

    std::cout << "========================================================================================\n";
    std::cout << " POMAI DB — KERNEL MICROBENCHMARK SUITE (Scalar Oracle vs SIMD)\n";
    std::cout << " Iterations: " << iters << " per test\n";
    std::cout << "========================================================================================\n\n";

    // 1. Inner Product / Dot
    std::cout << "--- Metric: Inner Product (Dot) ---\n";
    std::cout << std::setw(6) << "Dim"
              << std::setw(14) << "Scalar (ns)"
              << std::setw(14) << "SIMD (ns)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Throughput(Mv/s)"
              << std::setw(16) << "Bandwidth(GB/s)" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (size_t d : dims) {
        auto r = RunBench(d, iters,
            [](const auto& a, const auto& b) { return pomai::core::reference::InnerProduct(a, b); },
            [](const auto& a, const auto& b) { return pomai::core::Dot(a, b); });

        std::cout << std::setw(6) << d
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.scalar_ns
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.simd_ns
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.simd_mvec_s
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.simd_gb_s << "\n";
    }
    std::cout << "\n";

    // 2. L2 Squared
    std::cout << "--- Metric: Squared Euclidean (L2Sq) ---\n";
    std::cout << std::setw(6) << "Dim"
              << std::setw(14) << "Scalar (ns)"
              << std::setw(14) << "SIMD (ns)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Throughput(Mv/s)"
              << std::setw(16) << "Bandwidth(GB/s)" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (size_t d : dims) {
        auto r = RunBench(d, iters,
            [](const auto& a, const auto& b) { return pomai::core::reference::L2Sq(a, b); },
            [](const auto& a, const auto& b) { return pomai::core::L2Sq(a, b); });

        std::cout << std::setw(6) << d
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.scalar_ns
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.simd_ns
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.simd_mvec_s
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.simd_gb_s << "\n";
    }
    std::cout << "\n";

    // 3. Cosine Similarity
    std::cout << "--- Metric: Cosine Similarity ---\n";
    std::cout << std::setw(6) << "Dim"
              << std::setw(14) << "Scalar (ns)"
              << std::setw(14) << "SIMD (ns)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Throughput(Mv/s)"
              << std::setw(16) << "Bandwidth(GB/s)" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (size_t d : dims) {
        auto r = RunBench(d, iters,
            [](const auto& a, const auto& b) { return pomai::core::reference::CosineSimilarity(a, b); },
            [](const auto& a, const auto& b) { return pomai::core::CosineSimilarity(a, b); });

        std::cout << std::setw(6) << d
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.scalar_ns
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.simd_ns
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.simd_mvec_s
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.simd_gb_s << "\n";
    }
    std::cout << "\n";

    // 4. Quantized Distance: SQ8 L2
    std::cout << "--- Quantized Metric: SQ8 L2Sq ---\n";
    std::cout << std::setw(6) << "Dim"
              << std::setw(14) << "Scalar (ns)"
              << std::setw(14) << "SIMD (ns)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "Throughput(Mv/s)" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (size_t d : dims) {
        auto f_vec = MakeRandomVec(d, 555);
        std::vector<uint8_t> sq8_vec(d);
        for (size_t i = 0; i < d; ++i) sq8_vec[i] = static_cast<uint8_t>(i % 256);

        const auto t0 = Clock::now();
        volatile float d_scalar = 0.0f;
        for (size_t i = 0; i < iters; ++i) {
            d_scalar += static_cast<float>(pomai::core::reference::L2SqSq8(f_vec, sq8_vec, -1.0f, 1.0f));
        }
        const auto t1 = Clock::now();

        volatile float d_simd = 0.0f;
        const auto t2 = Clock::now();
        for (size_t i = 0; i < iters; ++i) {
            d_simd += pomai::core::L2SqSq8(f_vec, sq8_vec, -1.0f, 1.0f);
        }
        const auto t3 = Clock::now();

        double ns_scalar = (std::chrono::duration<double>(t1 - t0).count() / iters) * 1e9;
        double ns_simd = (std::chrono::duration<double>(t3 - t2).count() / iters) * 1e9;
        double speedup = ns_simd > 0.0 ? (ns_scalar / ns_simd) : 1.0;
        double mvec_s = ns_simd > 0.0 ? (1e3 / ns_simd) : 0.0;

        std::cout << std::setw(6) << d
                  << std::setw(14) << std::fixed << std::setprecision(1) << ns_scalar
                  << std::setw(14) << std::fixed << std::setprecision(1) << ns_simd
                  << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(16) << std::fixed << std::setprecision(2) << mvec_s << "\n";
    }
    std::cout << "\n";

    // 5. Top-K Selection: Full Sort vs Bounded Quickselect (SelectTopK)
    std::cout << "--- Top-K Selection: Full Sort vs Quickselect (N=5000, K=10) ---\n";
    const size_t N = 5000;
    const size_t K = 10;
    const size_t topk_iters = 5000;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);

    std::vector<pomai::core::TopKItem> base_items(N);
    for (size_t i = 0; i < N; ++i) {
        base_items[i].id = static_cast<pomai::VectorId>(i + 1);
        base_items[i].score = score_dist(rng);
    }

    // Full sort
    const auto t_sort_0 = Clock::now();
    for (size_t it = 0; it < topk_iters; ++it) {
        auto items = base_items;
        std::sort(items.begin(), items.end(), pomai::core::TopKScoreDescIdAsc{});
        items.resize(K);
    }
    const auto t_sort_1 = Clock::now();

    // Bounded SelectTopK
    const auto t_sel_0 = Clock::now();
    for (size_t it = 0; it < topk_iters; ++it) {
        auto items = base_items;
        pomai::core::SelectTopK(items, K);
    }
    const auto t_sel_1 = Clock::now();

    double full_sort_us = (std::chrono::duration<double>(t_sort_1 - t_sort_0).count() / topk_iters) * 1e6;
    double quickselect_us = (std::chrono::duration<double>(t_sel_1 - t_sel_0).count() / topk_iters) * 1e6;
    double topk_speedup = quickselect_us > 0.0 ? (full_sort_us / quickselect_us) : 1.0;

    std::cout << "Full std::sort (N=" << N << "): " << std::fixed << std::setprecision(2) << full_sort_us << " us\n";
    std::cout << "Bounded SelectTopK (K=" << K << "): " << std::fixed << std::setprecision(2) << quickselect_us << " us\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << topk_speedup << "x faster\n\n";

    return 0;
}
