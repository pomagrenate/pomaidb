#pragma once

// psync_telemetry.h
// Zero-cost latency & contention tracker using raw CPU cycles
// Target: Linux x86-64 / Windows
// Compilers: GCC, Clang, MSVC

#include "psync_platform.h"
#include <cstdio>

namespace psync {

// ============================================================================
// CPU CYCLE COUNTER (SERIALIZED)
// ============================================================================

// Read CPU cycle counter with serialization using rdtscp
// Prevents out-of-order execution corruption of cycle measurements
inline u64 rdtsc_serialized() {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: use __builtin_ia32_rdtscp for serialized read
    // The auxiliary parameter is required but we ignore it
    unsigned int aux;
    return __builtin_ia32_rdtscp(&aux);
#elif defined(_MSC_VER)
    // MSVC: use __rdtscp intrinsic (available on modern CPUs)
    unsigned int aux;
    return __rdtscp(&aux);
#else
    #error "Unsupported compiler for RDTSCP"
#endif
}

// Simple rdtsc without serialization (faster but may be reordered)
inline u64 rdtsc() {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: use __builtin_ia32_rdtsc
    return __builtin_ia32_rdtsc();
#elif defined(_MSC_VER)
    // MSVC: use __rdtsc intrinsic
    return __rdtsc();
#else
    #error "Unsupported compiler for RDTSC"
#endif
}

// ============================================================================
// TELEMETRY CONFIGURATION
// ============================================================================

// Compile-time enable/disable telemetry
#ifndef PSYNC_ENABLE_TELEMETRY
    #define PSYNC_ENABLE_TELEMETRY 0
#endif

// ============================================================================
// LOCK TELEMETRY DATA
// ============================================================================

struct LockTelemetry {
    u64 total_wait_cycles;
    u64 total_contention_count;
    u64 acquisition_count;
    u64 max_wait_cycles;
    
    constexpr LockTelemetry() 
        : total_wait_cycles(0), 
          total_contention_count(0), 
          acquisition_count(0),
          max_wait_cycles(0) {}
    
    void reset() {
        total_wait_cycles = 0;
        total_contention_count = 0;
        acquisition_count = 0;
        max_wait_cycles = 0;
    }
    
    double get_average_wait_cycles() const {
        if (total_contention_count == 0) return 0.0;
        return static_cast<double>(total_wait_cycles) / static_cast<double>(total_contention_count);
    }
    
    double get_contention_rate() const {
        if (acquisition_count == 0) return 0.0;
        return static_cast<double>(total_contention_count) / static_cast<double>(acquisition_count);
    }
};

// ============================================================================
// LOCK TELEMETRY COLLECTOR
// ============================================================================

class LockTelemetryCollector {
public:
    constexpr LockTelemetryCollector() : telemetry_(), start_time_(0) {}
    
    // Record lock acquisition start (uses serialized rdtscp for accuracy)
    void record_acquisition_start() {
        if constexpr (PSYNC_ENABLE_TELEMETRY) {
            start_time_ = rdtsc_serialized();
        }
    }
    
    // Record successful lock acquisition (no contention)
    void record_acquisition_success() {
        if constexpr (PSYNC_ENABLE_TELEMETRY) {
            atomic_fetch_add(&telemetry_.acquisition_count, static_cast<u64>(1), MemoryOrder::relaxed);
        }
    }
    
    // Record contended lock acquisition (uses serialized rdtscp for accuracy)
    void record_acquisition_contended() {
        if constexpr (PSYNC_ENABLE_TELEMETRY) {
            u64 end_time = rdtsc_serialized();
            u64 wait_cycles = end_time - start_time_;
            
            atomic_fetch_add(&telemetry_.total_wait_cycles, wait_cycles, MemoryOrder::relaxed);
            atomic_fetch_add(&telemetry_.total_contention_count, static_cast<u64>(1), MemoryOrder::relaxed);
            atomic_fetch_add(&telemetry_.acquisition_count, static_cast<u64>(1), MemoryOrder::relaxed);
            
            // Update max wait cycles (relaxed, best effort)
            u64 current_max = atomic_load(&telemetry_.max_wait_cycles, MemoryOrder::relaxed);
            while (wait_cycles > current_max) {
                u64 expected = current_max;
                if (atomic_compare_exchange(
                    &telemetry_.max_wait_cycles,
                    &expected,
                    wait_cycles,
                    MemoryOrder::relaxed,
                    MemoryOrder::relaxed
                )) {
                    break;
                }
                current_max = expected;
            }
        }
    }
    
    // Get telemetry data (relaxed read, no synchronization)
    LockTelemetry get_telemetry() const {
        if constexpr (PSYNC_ENABLE_TELEMETRY) {
            LockTelemetry result;
            result.total_wait_cycles = atomic_load(&telemetry_.total_wait_cycles, MemoryOrder::relaxed);
            result.total_contention_count = atomic_load(&telemetry_.total_contention_count, MemoryOrder::relaxed);
            result.acquisition_count = atomic_load(&telemetry_.acquisition_count, MemoryOrder::relaxed);
            result.max_wait_cycles = atomic_load(&telemetry_.max_wait_cycles, MemoryOrder::relaxed);
            return result;
        } else {
            return LockTelemetry();
        }
    }
    
    // Reset telemetry
    void reset() {
        if constexpr (PSYNC_ENABLE_TELEMETRY) {
            telemetry_.reset();
        }
    }
    
private:
    LockTelemetry telemetry_;
    u64 start_time_;
};

// ============================================================================
// TELEMETRY ENABLED LOCK WRAPPER
// ============================================================================

template<typename Lock>
class TelemetryLock {
public:
    constexpr TelemetryLock() : lock_(), telemetry_() {}
    
    void lock() {
        telemetry_.record_acquisition_start();
        
        // Try fast path
        if (lock_.try_lock()) {
            telemetry_.record_acquisition_success();
            return;
        }
        
        // Contended path
        lock_.lock();
        telemetry_.record_acquisition_contended();
    }
    
    bool try_lock() {
        telemetry_.record_acquisition_start();
        
        if (lock_.try_lock()) {
            telemetry_.record_acquisition_success();
            return true;
        }
        
        telemetry_.record_acquisition_contended();
        return false;
    }
    
    void unlock() {
        lock_.unlock();
    }
    
    // Get telemetry data
    LockTelemetry get_telemetry() const {
        return telemetry_.get_telemetry();
    }
    
    // Reset telemetry
    void reset_telemetry() {
        telemetry_.reset();
    }
    
    // Get telemetry collector (for registration)
    LockTelemetryCollector* get_telemetry_collector() {
        return &telemetry_;
    }
    
    // Get underlying lock (for advanced usage)
    Lock& get_lock() {
        return lock_;
    }
    
    const Lock& get_lock() const {
        return lock_;
    }
    
private:
    Lock lock_;
    LockTelemetryCollector telemetry_;
};

// ============================================================================
// GLOBAL TELEMETRY REGISTRY
// ============================================================================

class TelemetryRegistry {
public:
    static constexpr usize MAX_LOCKS = 64;
    
    struct LockEntry {
        const char* name;
        LockTelemetryCollector* collector;
        
        constexpr LockEntry() : name(nullptr), collector(nullptr) {}
        constexpr LockEntry(const char* n, LockTelemetryCollector* c) : name(n), collector(c) {}
    };
    
    static TelemetryRegistry& instance() {
        static TelemetryRegistry registry;
        return registry;
    }
    
    void register_lock(const char* name, LockTelemetryCollector* collector) {
        for (usize i = 0; i < MAX_LOCKS; ++i) {
            if (entries_[i].collector == nullptr) {
                entries_[i] = LockEntry(name, collector);
                return;
            }
        }
        // Registry full, ignore (in production, this would be an error)
    }
    
    void print_report() {
        if constexpr (PSYNC_ENABLE_TELEMETRY) {
            printf("=== LOCK TELEMETRY REPORT ===\n");
            for (usize i = 0; i < MAX_LOCKS; ++i) {
                if (entries_[i].collector != nullptr) {
                    LockTelemetry telemetry = entries_[i].collector->get_telemetry();
                    printf("Lock: %s\n", entries_[i].name);
                    printf("  Acquisitions: %llu\n", telemetry.acquisition_count);
                    printf("  Contentions: %llu\n", telemetry.total_contention_count);
                    printf("  Contention Rate: %.2f%%\n", telemetry.get_contention_rate() * 100.0);
                    printf("  Avg Wait Cycles: %.2f\n", telemetry.get_average_wait_cycles());
                    printf("  Max Wait Cycles: %llu\n", telemetry.max_wait_cycles);
                }
            }
        } else {
            printf("Telemetry disabled (compile-time flag PSYNC_ENABLE_TELEMETRY=0)\n");
        }
    }
    
private:
    constexpr TelemetryRegistry() : entries_() {}
    
    LockEntry entries_[MAX_LOCKS];
};

// ============================================================================
// TELEMETRY LOCK WITH REGISTRATION
// ============================================================================

template<typename Lock>
class RegisteredTelemetryLock : public TelemetryLock<Lock> {
public:
    explicit RegisteredTelemetryLock(const char* name) : TelemetryLock<Lock>(), name_(name) {
        TelemetryRegistry::instance().register_lock(name_, this->get_telemetry_collector());
    }
    
private:
    const char* name_;
};

// ============================================================================
// RAII TELEMETRY SCOPED TIMER
// ============================================================================

class ScopedTelemetryTimer {
public:
    explicit ScopedTelemetryTimer(LockTelemetryCollector* collector) 
        : collector_(collector), start_time_(rdtsc()) {}
    
    ~ScopedTelemetryTimer() {
        if (collector_ != nullptr) {
            u64 end_time = rdtsc();
            u64 elapsed = end_time - start_time_;
            LockTelemetry telemetry = collector_->get_telemetry();
            atomic_fetch_add(&telemetry.total_wait_cycles, elapsed, MemoryOrder::relaxed);
        }
    }
    
private:
    LockTelemetryCollector* collector_;
    u64 start_time_;
};

} // namespace psync