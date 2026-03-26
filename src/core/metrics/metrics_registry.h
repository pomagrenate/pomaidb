#pragma once
#include <atomic>
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace pomai::core::metrics {

    enum class MetricType {
        kCounter,
        kGauge
    };

    /**
     * Centralized Metrics Registry for PomaiDB.
     * Uses atomic counters for low-overhead instrumentation in the hot path.
     */
    class MetricsRegistry {
    public:
        static MetricsRegistry& Instance() {
            static MetricsRegistry instance;
            return instance;
        }

        void Increment(const std::string& name, uint64_t count = 1) {
            counters_[name].fetch_add(count, std::memory_order_relaxed);
        }

        void SetGauge(const std::string& name, uint64_t value) {
            gauges_[name].store(value, std::memory_order_relaxed);
        }

        uint64_t GetCounter(const std::string& name) const {
            auto it = counters_.find(name);
            return (it != counters_.end()) ? it->second.load(std::memory_order_relaxed) : 0;
        }

        uint64_t GetGauge(const std::string& name) const {
            auto it = gauges_.find(name);
            return (it != gauges_.end()) ? it->second.load(std::memory_order_relaxed) : 0;
        }

        std::map<std::string, uint64_t> SnapshotCounters() const {
            std::map<std::string, uint64_t> res;
            for (const auto& [name, val] : counters_) {
                res[name] = val.load(std::memory_order_relaxed);
            }
            return res;
        }

    private:
        MetricsRegistry() = default;
        mutable std::map<std::string, std::atomic<uint64_t>> counters_;
        mutable std::map<std::string, std::atomic<uint64_t>> gauges_;
    };

} // namespace pomai::core::metrics
