#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

#include <palloc.h>
#include <palloc_vector.h>
#include <palloc/arena_pomai.h>
#include "src/utils/palloc_compat.h"

namespace pomai::palloc_qa {

struct AllocationRecord {
    void* raw_ptr = nullptr;
    void* user_ptr = nullptr;
    size_t requested_size = 0;
    size_t usable_size = 0;
    size_t alignment = 0;
    size_t guard_size = 0;
    uint8_t poison_byte = 0;
    uint32_t payload_seed = 0;
    uint64_t alloc_id = 0;
    uint32_t thread_id = 0;
};

class AllocatorOracle {
public:
    AllocatorOracle(bool enable_guards = true, size_t guard_size = 64)
        : enable_guards_(enable_guards), guard_size_(guard_size) {}

    ~AllocatorOracle() {
        if (!live_allocs_.empty()) {
            std::cerr << "[ORACLE] Leaked " << live_allocs_.size() << " live allocations!" << std::endl;
        }
    }

    // O(log N) verification that [start, end) does not overlap any live interval
    void CheckNoOverlap(uintptr_t start, uintptr_t end) const {
        if (live_intervals_.empty()) return;

        auto it = live_intervals_.upper_bound(start);
        if (it != live_intervals_.end() && it->first < end) {
            std::ostringstream ss;
            ss << "OVERLAPPING ALLOCATIONS DETECTED: ["
               << std::hex << start << ", " << end << ") intersects with ["
               << it->first << ", " << it->second << ")";
            throw std::runtime_error(ss.str());
        }
        if (it != live_intervals_.begin()) {
            auto prev = std::prev(it);
            if (prev->second > start) {
                std::ostringstream ss;
                ss << "OVERLAPPING ALLOCATIONS DETECTED: ["
                   << std::hex << start << ", " << end << ") intersects with ["
                   << prev->first << ", " << prev->second << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }

    void* TrackAlloc(size_t size, size_t alignment = 16, uint32_t seed = 0, uint32_t tid = 0) {
        if (alignment == 0) alignment = 16;
        size_t actual_guard = enable_guards_ ? guard_size_ : 0;
        size_t total_alloc = actual_guard + size + actual_guard;
        if (total_alloc < size) return nullptr; // overflow

        void* raw = nullptr;
        if (alignment > 0) {
            raw = pa_malloc_aligned(total_alloc, alignment);
        } else {
            raw = pa_malloc(total_alloc);
        }
        if (!raw) return nullptr;

        size_t usable = pa_usable_size(raw);
        if (usable < total_alloc) {
            pa_free(raw);
            throw std::runtime_error("pa_usable_size < requested total_alloc!");
        }

        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
        CheckNoOverlap(raw_addr, raw_addr + usable);

        uint8_t* user_ptr = static_cast<uint8_t*>(raw) + actual_guard;

        uint8_t poison = static_cast<uint8_t>(0xA5 ^ (seed & 0xFF));
        if (actual_guard > 0) {
            std::memset(raw, poison, actual_guard);
            std::memset(user_ptr + size, poison, actual_guard);
        }

        // Fill payload
        FillPayload(user_ptr, size, seed);

        uint64_t id = ++alloc_counter_;
        AllocationRecord rec;
        rec.raw_ptr = raw;
        rec.user_ptr = user_ptr;
        rec.requested_size = size;
        rec.usable_size = usable;
        rec.alignment = alignment;
        rec.guard_size = actual_guard;
        rec.poison_byte = poison;
        rec.payload_seed = seed;
        rec.alloc_id = id;
        rec.thread_id = tid;

        live_allocs_[user_ptr] = rec;
        live_intervals_[raw_addr] = raw_addr + usable;
        return user_ptr;
    }

    bool VerifyAndTouch(void* user_ptr) {
        auto it = live_allocs_.find(user_ptr);
        if (it == live_allocs_.end()) return false;
        const auto& rec = it->second;

        // Verify front guard
        if (rec.guard_size > 0) {
            const uint8_t* front = static_cast<const uint8_t*>(rec.raw_ptr);
            for (size_t i = 0; i < rec.guard_size; ++i) {
                if (front[i] != rec.poison_byte) {
                    throw std::runtime_error("Front guard red-zone corruption detected!");
                }
            }
            // Verify rear guard
            const uint8_t* rear = static_cast<const uint8_t*>(rec.user_ptr) + rec.requested_size;
            for (size_t i = 0; i < rec.guard_size; ++i) {
                if (rear[i] != rec.poison_byte) {
                    throw std::runtime_error("Rear guard red-zone corruption detected!");
                }
            }
        }

        // Verify payload
        VerifyPayload(static_cast<const uint8_t*>(rec.user_ptr), rec.requested_size, rec.payload_seed);
        return true;
    }

    bool TrackFree(void* user_ptr) {
        auto it = live_allocs_.find(user_ptr);
        if (it == live_allocs_.end()) {
            throw std::runtime_error("Attempted to free untracked/already-freed pointer!");
        }
        VerifyAndTouch(user_ptr);
        void* raw = it->second.raw_ptr;
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);

        live_intervals_.erase(raw_addr);
        live_allocs_.erase(it);

        pa_free(raw);
        return true;
    }

    size_t LiveCount() const { return live_allocs_.size(); }

private:
    static void FillPayload(uint8_t* p, size_t size, uint32_t seed) {
        uint32_t state = seed == 0 ? 0x12345678 : seed;
        for (size_t i = 0; i < size; ++i) {
            state = state * 1664525u + 1013904223u;
            p[i] = static_cast<uint8_t>(state >> 24);
        }
    }

    static void VerifyPayload(const uint8_t* p, size_t size, uint32_t seed) {
        uint32_t state = seed == 0 ? 0x12345678 : seed;
        for (size_t i = 0; i < size; ++i) {
            state = state * 1664525u + 1013904223u;
            uint8_t expected = static_cast<uint8_t>(state >> 24);
            if (p[i] != expected) {
                std::ostringstream ss;
                ss << "Payload corruption at byte offset " << i
                   << "! expected: 0x" << std::hex << (int)expected
                   << " actual: 0x" << (int)p[i];
                throw std::runtime_error(ss.str());
            }
        }
    }

    bool enable_guards_ = true;
    size_t guard_size_ = 64;
    uint64_t alloc_counter_ = 0;
    std::unordered_map<void*, AllocationRecord> live_allocs_;
    std::map<uintptr_t, uintptr_t> live_intervals_;
};

} // namespace pomai::palloc_qa
