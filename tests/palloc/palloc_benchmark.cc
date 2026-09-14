#include "tests/common/test_main.h"

#include <palloc.h>
#include <palloc_vector.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace pomai::palloc_qa {

struct LatencyHistogram {
    std::vector<uint64_t> samples_ns;

    void Record(uint64_t ns) {
        samples_ns.push_back(ns);
    }

    void PrintSummary(const char* label, uint64_t total_ops, double total_time_sec) {
        if (samples_ns.empty()) return;
        std::sort(samples_ns.begin(), samples_ns.end());
        size_t n = samples_ns.size();

        uint64_t p50 = samples_ns[n * 50 / 100];
        uint64_t p95 = samples_ns[n * 95 / 100];
        uint64_t p99 = samples_ns[n * 99 / 100];
        uint64_t p999 = samples_ns[n * 999 / 1000];
        double mean_ns = (total_time_sec * 1e9) / total_ops;
        double ops_per_sec = total_ops / total_time_sec;

        std::cout << std::left << std::setw(20) << label
                  << " | " << std::right << std::setw(10) << static_cast<uint64_t>(ops_per_sec) << " ops/s"
                  << " | mean: " << std::setw(6) << std::fixed << std::setprecision(1) << mean_ns << " ns"
                  << " | p50: " << std::setw(5) << p50 << " ns"
                  << " | p95: " << std::setw(5) << p95 << " ns"
                  << " | p99: " << std::setw(5) << p99 << " ns"
                  << " | p99.9: " << std::setw(5) << p999 << " ns\n";
    }
};

POMAI_TEST(PallocBenchmark_VersusSystemMalloc) {
    constexpr int kIterations = 100000;
    const size_t kDims[] = {32, 128, 512, 1536}; // vector float counts

    std::cout << "\n================ ALLOCATOR BENCHMARK: PALLOC VS SYSTEM MALLOC ================\n";

    // 1. Single-threaded Vector Allocation Benchmark (palloc)
    {
        LatencyHistogram hist;
        hist.samples_ns.reserve(kIterations);

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIterations; ++i) {
            size_t dim = kDims[i % 4];
            auto s0 = std::chrono::steady_clock::now();
            float* p = pa_vector_alloc_floats(dim);
            p[0] = 1.0f;
            p[dim - 1] = 2.0f;
            pa_free(p);
            auto s1 = std::chrono::steady_clock::now();
            hist.Record(std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count());
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        hist.PrintSummary("palloc (ST, aligned)", kIterations, elapsed);
    }

    // 2. Single-threaded Vector Allocation Benchmark (system malloc)
    {
        LatencyHistogram hist;
        hist.samples_ns.reserve(kIterations);

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIterations; ++i) {
            size_t dim = kDims[i % 4];
            size_t bytes = dim * sizeof(float);
            auto s0 = std::chrono::steady_clock::now();
            void* raw = std::malloc(bytes);
            float* p = static_cast<float*>(raw);
            p[0] = 1.0f;
            p[dim - 1] = 2.0f;
            std::free(raw);
            auto s1 = std::chrono::steady_clock::now();
            hist.Record(std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count());
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        hist.PrintSummary("system malloc (ST)", kIterations, elapsed);
    }

    // 3. Multi-threaded Vector Allocation Benchmark (palloc, 8 threads)
    {
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 25000;
        auto t0 = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < kOpsPerThread; ++i) {
                    size_t dim = kDims[(i + t) % 4];
                    float* p = pa_vector_alloc_floats(dim);
                    p[0] = static_cast<float>(t);
                    p[dim - 1] = 1.0f;
                    pa_free(p);
                }
            });
        }
        for (auto& th : threads) th.join();
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double ops_per_sec = (kThreads * kOpsPerThread) / elapsed;
        std::cout << std::left << std::setw(20) << "palloc (8T, aligned)"
                  << " | " << std::right << std::setw(10) << static_cast<uint64_t>(ops_per_sec) << " ops/s"
                  << " | throughput: " << std::fixed << std::setprecision(2) << (ops_per_sec / 1e6) << " M ops/s\n";
    }

    // 4. Multi-threaded Vector Allocation Benchmark (system malloc, 8 threads)
    {
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 25000;
        auto t0 = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < kOpsPerThread; ++i) {
                    size_t dim = kDims[(i + t) % 4];
                    void* raw = std::malloc(dim * sizeof(float));
                    float* p = static_cast<float*>(raw);
                    p[0] = static_cast<float>(t);
                    p[dim - 1] = 1.0f;
                    std::free(raw);
                }
            });
        }
        for (auto& th : threads) th.join();
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double ops_per_sec = (kThreads * kOpsPerThread) / elapsed;
        std::cout << std::left << std::setw(20) << "system malloc (8T)"
                  << " | " << std::right << std::setw(10) << static_cast<uint64_t>(ops_per_sec) << " ops/s"
                  << " | throughput: " << std::fixed << std::setprecision(2) << (ops_per_sec / 1e6) << " M ops/s\n";
    }

    std::cout << "===============================================================================\n\n";
}

} // namespace pomai::palloc_qa
