// benchmarks/round3_rind_concurrency_fix.cc — Rind Read Concurrency Optimization (Section 16)
//
// Quantifies:
// 1. Current production Rind: Every vector query executes rind->IsDeleted(id) taking mu_
// 2. Optimized Rind: Query takes a single point-in-time deleted set snapshot at start, 0 locks in scan loop
// Across 1, 2, 4, 8, 16 concurrent threads in native C++.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

struct ConcurrencyResult {
    uint32_t threads;
    double aggregate_qps;
    double p50_ms;
    double p95_ms;
    double efficiency;
};

double Percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

class SimulatedRind {
public:
    mutable std::mutex mu_;
    std::unordered_set<uint64_t> tombstones_;

    SimulatedRind() {
        // Insert a few deleted vector IDs
        for (uint64_t i = 50; i < 70; ++i) tombstones_.insert(i);
    }

    // Flawed production path: Acquires mutex on EVERY vector check
    bool IsDeletedFlawed(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mu_);
        return tombstones_.count(id) > 0;
    }

    // Optimized snapshot path: Takes snapshot ONCE at query start
    std::unordered_set<uint64_t> GetTombstoneSnapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return tombstones_;
    }
};

ConcurrencyResult BenchmarkConcurrency(const SimulatedRind& rind,
                                       bool use_snapshot,
                                       uint32_t num_threads,
                                       uint32_t queries_per_thread,
                                       size_t n,
                                       double baseline_qps_1) {
    std::atomic<bool> start_gate{false};
    std::vector<std::thread> workers;
    std::vector<std::vector<double>> thread_lats(num_threads);
    for (auto& tl : thread_lats) tl.reserve(queries_per_thread);

    for (uint32_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            while (!start_gate.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (uint32_t q = 0; q < queries_per_thread; ++q) {
                auto t0 = std::chrono::high_resolution_clock::now();

                if (use_snapshot) {
                    // Optimized: Snapshot taken once per query
                    auto snap = rind.GetTombstoneSnapshot();
                    size_t live_count = 0;
                    for (size_t i = 0; i < n; ++i) {
                        if (snap.count(i)) continue;
                        live_count++;
                    }
                } else {
                    // Flawed: Mutex locked on EVERY vector
                    size_t live_count = 0;
                    for (size_t i = 0; i < n; ++i) {
                        if (rind.IsDeletedFlawed(i)) continue;
                        live_count++;
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                thread_lats[t].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        });
    }

    auto wall_start = std::chrono::high_resolution_clock::now();
    start_gate.store(true, std::memory_order_release);
    for (auto& w : workers) w.join();
    auto wall_end = std::chrono::high_resolution_clock::now();

    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();
    uint64_t total_queries = static_cast<uint64_t>(num_threads) * queries_per_thread;
    double qps = total_queries / wall_s;

    std::vector<double> all_lats;
    for (const auto& tl : thread_lats) all_lats.insert(all_lats.end(), tl.begin(), tl.end());

    double p50 = Percentile(all_lats, 0.50) / 1000.0;
    double p95 = Percentile(all_lats, 0.95) / 1000.0;
    double eff = (baseline_qps_1 > 0.0) ? (qps / (num_threads * baseline_qps_1)) * 100.0 : 100.0;

    return {num_threads, qps, p50, p95, eff};
}

} // namespace

int main() {
    std::cout << "=================================================================\n";
    std::cout << " PomaiDB Performance Round 3: Rind Concurrency Fix (Section 16) \n";
    std::cout << "=================================================================\n\n";

    const size_t n = 10000;
    const uint32_t queries_per_thread = 100;
    const std::vector<uint32_t> thread_counts = {1, 2, 4, 8, 16};
    SimulatedRind rind;

    std::cout << "### 16.1 CURRENT FLAWED RIND (Mutex lock per vector, N = 10,000)\n\n";
    std::cout << "| Threads | Aggregate QPS | Latency p50 | Latency p95 | Scaling Efficiency |\n";
    std::cout << "| :---: | :---: | :---: | :---: | :---: |\n";

    double flawed_base = 0.0;
    for (uint32_t t : thread_counts) {
        auto res = BenchmarkConcurrency(rind, false, t, queries_per_thread, n, flawed_base);
        if (t == 1) flawed_base = res.aggregate_qps;

        std::cout << "| " << std::setw(7) << res.threads
                  << " | " << std::fixed << std::setprecision(1) << std::setw(13) << res.aggregate_qps
                  << " | " << std::fixed << std::setprecision(2) << std::setw(9) << res.p50_ms << " ms"
                  << " | " << std::fixed << std::setprecision(2) << std::setw(9) << res.p95_ms << " ms"
                  << " | " << std::fixed << std::setprecision(1) << std::setw(16) << res.efficiency << "% |\n";
    }
    std::cout << "\n";

    std::cout << "### 16.2 OPTIMIZED RIND (Single snapshot at query start, N = 10,000)\n\n";
    std::cout << "| Threads | Aggregate QPS | Latency p50 | Latency p95 | Scaling Efficiency | Speedup over Flawed |\n";
    std::cout << "| :---: | :---: | :---: | :---: | :---: | :---: |\n";

    double opt_base = 0.0;
    for (uint32_t t : thread_counts) {
        auto res = BenchmarkConcurrency(rind, true, t, queries_per_thread, n, opt_base);
        if (t == 1) opt_base = res.aggregate_qps;

        std::cout << "| " << std::setw(7) << res.threads
                  << " | " << std::fixed << std::setprecision(1) << std::setw(13) << res.aggregate_qps
                  << " | " << std::fixed << std::setprecision(2) << std::setw(9) << res.p50_ms << " ms"
                  << " | " << std::fixed << std::setprecision(2) << std::setw(9) << res.p95_ms << " ms"
                  << " | " << std::fixed << std::setprecision(1) << std::setw(16) << res.efficiency << "%"
                  << " | **" << std::fixed << std::setprecision(1) << (res.aggregate_qps / flawed_base) << "×** |\n";
    }
    std::cout << "\n";

    return 0;
}
