#pragma once
#include <cstdint>
#include <vector>
#include "types.h"
#include "flat_hash_memmap.h"

namespace pomai::core {

/// SeenTracker: Runtime-local reusable structure for tracking seen IDs during search.
/// This replaces per-query unordered_set allocations with a reusable structure.
/// 
/// Design: Use a generation-based approach to avoid clearing the entire structure.
/// Each search increments the generation. An ID is "seen" if its stored generation
/// matches the current generation.
class SeenTracker {
public:
    SeenTracker() = default;

    /// Start a new search iteration. Increments generation.
    void BeginSearch() {
        ++current_generation_;
        // Handle overflow: reset all entries (rare, only every 2^32 searches)
        if (current_generation_ == 0) {
            entries_.Clear();
            current_generation_ = 1;
        }
    }

    /// Check if ID has been seen in current search.
    bool Contains(VectorId id) const {
        const auto* e = entries_.Find(id);
        if (!e) return false;
        return e->generation == current_generation_;
    }

    /// Mark ID as seen in current search.
    void MarkSeen(VectorId id) {
        entries_.Put(id, {current_generation_, false});
    }

    /// Mark ID as tombstone in current search.
    void MarkTombstone(VectorId id) {
        entries_.Put(id, {current_generation_, true});
    }

    /// Check if ID is tombstone in current search.
    bool IsTombstone(VectorId id) const {
        const auto* e = entries_.Find(id);
        if (!e || e->generation != current_generation_) return false;
        return e->is_tombstone;
    }

    /// Reserve capacity (called once at initialization).
    void Reserve(size_t capacity) {
        entries_.reserve(capacity);
    }

private:
    struct Entry {
        uint32_t generation;
        bool is_tombstone;
    };

    table::FlatHashMemMap<VectorId, Entry> entries_;
    uint32_t current_generation_ = 0;
};

} // namespace pomai::core
