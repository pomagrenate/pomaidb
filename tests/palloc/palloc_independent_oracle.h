// PALLOC INDEPENDENT SHADOW ORACLE — Hostile Forensic Verification
// Rule 1: This oracle does NOT use palloc for its own accounting.
// All internal tracking uses system malloc (::malloc / ::free) directly.
// This breaks circular validation: palloc cannot hide from its own shadow model.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <atomic>
#include <mutex>
#include <map>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <palloc.h>
#include <palloc_vector.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace pomai::palloc_forensic {

// ------------------------------------------------------------------
// System allocator baseline — completely independent of palloc.
// Used for oracle's own internal containers.
// ------------------------------------------------------------------
template<typename T>
struct SysAllocator {
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template<typename U>
    struct rebind { using other = SysAllocator<U>; };

    SysAllocator() noexcept = default;
    template<typename U>
    SysAllocator(const SysAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p = ::malloc(n * sizeof(T));
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { ::free(p); }

    template<typename U>
    bool operator==(const SysAllocator<U>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const SysAllocator<U>&) const noexcept { return false; }
};

using SysString = std::basic_string<char, std::char_traits<char>, SysAllocator<char>>;

template<typename K, typename V>
using SysMap = std::map<K, V, std::less<K>,
    SysAllocator<std::pair<const K, V>>>;

// Use SysMap (ordered) for all shadow tracking — avoids unordered_map
// rebind complexity while still providing O(log N) lookups.
// The oracle correctness proof does NOT depend on O(1) hash lookups.
template<typename K, typename V>
using SysUMap = std::map<K, V, std::less<K>,
    SysAllocator<std::pair<const K, V>>>;

template<typename T>
using SysVec = std::vector<T, SysAllocator<T>>;

// ------------------------------------------------------------------
// Fragmentation Measurement — Independent of palloc statistics
// ------------------------------------------------------------------
struct FragmentationReport {
    size_t total_requested_bytes    = 0;  // sum of all TrackAlloc sizes
    size_t total_usable_bytes       = 0;  // sum of pa_usable_size returns
    size_t live_requested_bytes     = 0;
    size_t live_usable_bytes        = 0;
    size_t live_count               = 0;
    size_t peak_live_count          = 0;
    size_t total_allocs             = 0;
    size_t total_frees              = 0;
    size_t alignment_violations     = 0;
    size_t overlap_detections       = 0;
    size_t canary_corruptions       = 0;
    size_t double_free_attempts     = 0;
    size_t invalid_free_attempts    = 0;

    // Internal fragmentation = (usable - requested) / requested
    double internal_frag_ratio() const {
        if (total_requested_bytes == 0) return 0.0;
        return double(total_usable_bytes - total_requested_bytes) / total_requested_bytes;
    }
};

// ------------------------------------------------------------------
// Process Memory Snapshot (OS-independent of palloc)
// ------------------------------------------------------------------
struct OsMemSnapshot {
    uint64_t timestamp_us       = 0;
    size_t   working_set_bytes  = 0;
    size_t   private_bytes      = 0;
    size_t   committed_bytes    = 0;
    size_t   reserved_virt      = 0;
    bool     valid              = false;

    static OsMemSnapshot capture() {
        OsMemSnapshot s;
        s.timestamp_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        pmc.cb = sizeof(pmc);
        if (::GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                sizeof(pmc))) {
            s.working_set_bytes = pmc.WorkingSetSize;
            s.private_bytes     = pmc.PrivateUsage;
            s.committed_bytes   = pmc.PrivateUsage;
            s.valid             = true;
        }
#endif
        return s;
    }
};

// ------------------------------------------------------------------
// AllocationRecord — stored with system malloc, never in palloc heap
// ------------------------------------------------------------------
struct AllocRecord {
    uintptr_t raw_addr      = 0;
    uintptr_t user_addr     = 0;
    size_t    req_size      = 0;
    size_t    usable_size   = 0;
    size_t    alignment     = 0;
    size_t    guard_front   = 0;
    size_t    guard_back    = 0;
    uint8_t   poison        = 0;
    uint32_t  payload_seed  = 0;
    uint64_t  alloc_id      = 0;
    uint32_t  thread_id     = 0;
    bool      freed         = false;
};

// ------------------------------------------------------------------
// Canary patterns — deliberately different from palloc's patterns
// ------------------------------------------------------------------
static constexpr uint8_t kFrontCanary  = 0xFE;  // Front guard sentinel
static constexpr uint8_t kBackCanary   = 0xEF;   // Back guard sentinel
static constexpr uint8_t kFreePoison   = 0xDD;   // Post-free poison

// ------------------------------------------------------------------
// IndependentOracle — the heart of the forensic audit
// All containers use system malloc via SysAllocator.
// palloc is called only for target allocations under test.
// ------------------------------------------------------------------
class IndependentOracle {
public:
    static constexpr size_t kDefaultGuard = 64;
    static constexpr size_t kMinGuard     = 16;
    static constexpr size_t kMaxGuard     = 256;

    explicit IndependentOracle(size_t guard_bytes = kDefaultGuard)
        : guard_bytes_(guard_bytes < kMinGuard ? kMinGuard :
                       guard_bytes > kMaxGuard ? kMaxGuard : guard_bytes) {}

    ~IndependentOracle() {
        if (!live_by_user_.empty()) {
            std::cerr << "[FORENSIC-ORACLE] LEAK: " << live_by_user_.size()
                      << " unfreed allocations!\n";
            frag_.canary_corruptions += 0; // count leaked as ok for dtor
        }
    }

    // ---------------------------------------------------------------
    // TrackAlloc: wraps palloc allocation with independent canaries.
    // Returns user pointer (same usage as original oracle),
    // but backing intervals stored in system-malloc containers.
    // ---------------------------------------------------------------
    void* TrackAlloc(size_t req_size, size_t alignment = 64,
                     uint32_t seed = 0, uint32_t tid = 0) {
        if (alignment == 0) alignment = 16;
        if ((alignment & (alignment - 1)) != 0) return nullptr; // must be pow2

        size_t total = guard_bytes_ + req_size + guard_bytes_;
        if (total < req_size) return nullptr; // overflow

        // Call palloc under test
        void* raw = nullptr;
        if (alignment > 0) {
            raw = pa_malloc_aligned(total, alignment);
        }
        if (!raw) return nullptr;

        // Independently verify alignment
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
        if ((raw_addr % alignment) != 0) {
            ++frag_.alignment_violations;
            pa_free(raw);
            std::ostringstream ss;
            ss << "ALIGNMENT VIOLATION: ptr 0x" << std::hex << raw_addr
               << " not aligned to " << alignment;
            throw std::runtime_error(ss.str());
        }

        // Verify pa_usable_size independently
        size_t usable = pa_usable_size(raw);
        if (usable < total) {
            pa_free(raw);
            throw std::runtime_error("pa_usable_size < total requested (guard+payload+guard)");
        }

        // Check no overlap with any live allocation
        {
            std::lock_guard<std::mutex> lk(mutex_);
            check_no_overlap_locked(raw_addr, raw_addr + usable);
        }

        // Paint front canary
        uint8_t* p8 = static_cast<uint8_t*>(raw);
        ::memset(p8, kFrontCanary, guard_bytes_);

        // Fill payload with deterministic PRNG (seed-based)
        uint8_t* payload = p8 + guard_bytes_;
        fill_payload(payload, req_size, seed);

        // Paint back canary
        ::memset(payload + req_size, kBackCanary, guard_bytes_);

        uint64_t id;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            id = ++alloc_counter_;

            AllocRecord rec;
            rec.raw_addr    = raw_addr;
            rec.user_addr   = reinterpret_cast<uintptr_t>(payload);
            rec.req_size    = req_size;
            rec.usable_size = usable;
            rec.alignment   = alignment;
            rec.guard_front = guard_bytes_;
            rec.guard_back  = guard_bytes_;
            rec.poison      = kFrontCanary;
            rec.payload_seed = seed;
            rec.alloc_id    = id;
            rec.thread_id   = tid;
            rec.freed       = false;

            live_by_user_[rec.user_addr] = rec;
            live_intervals_[raw_addr] = raw_addr + usable;

            frag_.total_allocs++;
            frag_.total_requested_bytes += req_size;
            frag_.total_usable_bytes    += usable;
            frag_.live_requested_bytes  += req_size;
            frag_.live_usable_bytes     += usable;
            frag_.live_count++;
            if (frag_.live_count > frag_.peak_live_count)
                frag_.peak_live_count = frag_.live_count;
        }

        return payload;
    }

    // ---------------------------------------------------------------
    // TrackFree: verify canaries + payload before freeing.
    // ---------------------------------------------------------------
    bool TrackFree(void* user_ptr) {
        if (!user_ptr) {
            ++frag_.invalid_free_attempts;
            throw std::runtime_error("TrackFree called with nullptr");
        }
        uintptr_t user_addr = reinterpret_cast<uintptr_t>(user_ptr);

        AllocRecord rec;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            auto it = live_by_user_.find(user_addr);
            if (it == live_by_user_.end()) {
                ++frag_.double_free_attempts;
                std::ostringstream ss;
                ss << "DOUBLE-FREE or INVALID-FREE: ptr 0x" << std::hex << user_addr;
                throw std::runtime_error(ss.str());
            }
            rec = it->second;
        }

        // Verify canaries (independent of palloc)
        verify_canaries(rec);

        // Verify payload
        verify_payload(static_cast<const uint8_t*>(user_ptr),
                       rec.req_size, rec.payload_seed);

        {
            std::lock_guard<std::mutex> lk(mutex_);
            live_intervals_.erase(rec.raw_addr);
            live_by_user_.erase(user_addr);

            frag_.total_frees++;
            frag_.live_requested_bytes -= rec.req_size;
            frag_.live_usable_bytes    -= rec.usable_size;
            frag_.live_count--;
        }

        // Poison back canary regions to catch use-after-free by caller
        ::memset(reinterpret_cast<void*>(rec.raw_addr), kFreePoison,
                 rec.guard_front + rec.req_size + rec.guard_back);

        pa_free(reinterpret_cast<void*>(rec.raw_addr));
        return true;
    }

    // ---------------------------------------------------------------
    // VerifyLive: verify a specific live allocation without freeing.
    // ---------------------------------------------------------------
    bool VerifyLive(void* user_ptr) {
        if (!user_ptr) return false;
        uintptr_t user_addr = reinterpret_cast<uintptr_t>(user_ptr);
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = live_by_user_.find(user_addr);
        if (it == live_by_user_.end()) return false;
        verify_canaries(it->second);
        verify_payload(static_cast<const uint8_t*>(user_ptr),
                       it->second.req_size, it->second.payload_seed);
        return true;
    }

    // ---------------------------------------------------------------
    // Free all remaining live allocations (for teardown).
    // ---------------------------------------------------------------
    void DrainAll() {
        // Collect all user pointers first (avoid iterator invalidation)
        SysVec<uintptr_t> addrs;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            for (auto& kv : live_by_user_) addrs.push_back(kv.first);
        }
        for (uintptr_t addr : addrs) {
            TrackFree(reinterpret_cast<void*>(addr));
        }
    }

    size_t live_count() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return frag_.live_count;
    }

    FragmentationReport report() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return frag_;
    }

    void print_report(std::ostream& os) const {
        auto r = report();
        os << "\n=== PALLOC INDEPENDENT ORACLE REPORT ===\n"
           << "  Total Allocs:          " << r.total_allocs << "\n"
           << "  Total Frees:           " << r.total_frees << "\n"
           << "  Live Count:            " << r.live_count << "\n"
           << "  Peak Live Count:       " << r.peak_live_count << "\n"
           << "  Total Requested Bytes: " << r.total_requested_bytes << "\n"
           << "  Total Usable Bytes:    " << r.total_usable_bytes << "\n"
           << "  Internal Frag Ratio:   "
           << r.internal_frag_ratio() * 100.0 << "%\n"
           << "  Alignment Violations:  " << r.alignment_violations << "\n"
           << "  Overlap Detections:    " << r.overlap_detections << "\n"
           << "  Canary Corruptions:    " << r.canary_corruptions << "\n"
           << "  Double Free Attempts:  " << r.double_free_attempts << "\n"
           << "===========================================\n";
    }

private:
    size_t guard_bytes_;
    mutable std::mutex mutex_;
    uint64_t alloc_counter_ = 0;
    FragmentationReport frag_;

    // Key insight: these containers use SysAllocator (system malloc).
    // palloc cannot affect oracle's own metadata.
    SysMap<uintptr_t, uintptr_t> live_intervals_;
    SysUMap<uintptr_t, AllocRecord> live_by_user_;

    // O(log N) overlap check (requires lock held)
    void check_no_overlap_locked(uintptr_t start, uintptr_t end) {
        auto it = live_intervals_.upper_bound(start);
        if (it != live_intervals_.end() && it->first < end) {
            ++frag_.overlap_detections;
            std::ostringstream ss;
            ss << "OVERLAP DETECTED [0x" << std::hex << start
               << ",0x" << end << ") vs [0x" << it->first
               << ",0x" << it->second << ")";
            throw std::runtime_error(ss.str());
        }
        if (it != live_intervals_.begin()) {
            auto prev = std::prev(it);
            if (prev->second > start) {
                ++frag_.overlap_detections;
                std::ostringstream ss;
                ss << "OVERLAP DETECTED [0x" << std::hex << start
                   << ",0x" << end << ") vs [0x" << prev->first
                   << ",0x" << prev->second << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }

    void verify_canaries(const AllocRecord& rec) const {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(rec.raw_addr);
        // Front canary
        for (size_t i = 0; i < rec.guard_front; ++i) {
            if (p[i] != kFrontCanary) {
                std::ostringstream ss;
                ss << "FRONT CANARY CORRUPTION at alloc_id=" << rec.alloc_id
                   << " offset=" << i << " expected=0x" << std::hex
                   << (int)kFrontCanary << " got=0x" << (int)p[i];
                throw std::runtime_error(ss.str());
            }
        }
        // Back canary
        const uint8_t* back = p + rec.guard_front + rec.req_size;
        for (size_t i = 0; i < rec.guard_back; ++i) {
            if (back[i] != kBackCanary) {
                std::ostringstream ss;
                ss << "BACK CANARY CORRUPTION at alloc_id=" << rec.alloc_id
                   << " offset=" << i << " expected=0x" << std::hex
                   << (int)kBackCanary << " got=0x" << (int)back[i];
                throw std::runtime_error(ss.str());
            }
        }
    }

    static void fill_payload(uint8_t* p, size_t size, uint32_t seed) {
        // Xorshift32 — deterministic, different from palloc's internal patterns
        uint32_t x = (seed == 0) ? 0xDEADBEEF : seed;
        for (size_t i = 0; i < size; ++i) {
            x ^= (x << 13); x ^= (x >> 17); x ^= (x << 5);
            p[i] = static_cast<uint8_t>(x & 0xFF);
        }
    }

    static void verify_payload(const uint8_t* p, size_t size, uint32_t seed) {
        uint32_t x = (seed == 0) ? 0xDEADBEEF : seed;
        for (size_t i = 0; i < size; ++i) {
            x ^= (x << 13); x ^= (x >> 17); x ^= (x << 5);
            uint8_t expected = static_cast<uint8_t>(x & 0xFF);
            if (p[i] != expected) {
                std::ostringstream ss;
                ss << "PAYLOAD CORRUPTION at byte=" << i
                   << " expected=0x" << std::hex << (int)expected
                   << " got=0x" << (int)p[i];
                throw std::runtime_error(ss.str());
            }
        }
    }
};

// ------------------------------------------------------------------
// Resource budget controller — prevents destroying the host machine.
// ------------------------------------------------------------------
class ResourceBudget {
public:
    explicit ResourceBudget(size_t max_bytes = 256 * 1024 * 1024 /* 256 MiB */)
        : max_bytes_(max_bytes), used_bytes_(0) {}

    // Returns true if we can afford this allocation.
    bool can_alloc(size_t bytes) const {
        return used_bytes_.load(std::memory_order_relaxed) + bytes <= max_bytes_;
    }

    void record_alloc(size_t bytes) {
        used_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }

    void record_free(size_t bytes) {
        size_t cur = used_bytes_.load(std::memory_order_relaxed);
        if (cur >= bytes)
            used_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
        else
            used_bytes_.store(0, std::memory_order_relaxed);
    }

    size_t used() const { return used_bytes_.load(std::memory_order_relaxed); }
    size_t max()  const { return max_bytes_; }

    // Check available OS memory and adapt budget down if necessary.
    static size_t safe_budget_bytes() {
#if defined(_WIN32)
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            // Use at most 30% of available physical RAM
            size_t avail = static_cast<size_t>(ms.ullAvailPhys);
            size_t budget = avail * 30 / 100;
            // Hard cap at 512 MiB
            if (budget > 512ULL * 1024 * 1024)
                budget = 512ULL * 1024 * 1024;
            // Minimum: 64 MiB
            if (budget < 64ULL * 1024 * 1024)
                budget = 64ULL * 1024 * 1024;
            return budget;
        }
#endif
        return 256ULL * 1024 * 1024; // conservative default
    }

private:
    size_t max_bytes_;
    std::atomic<size_t> used_bytes_;
};

} // namespace pomai::palloc_forensic
