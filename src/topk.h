// topk.h — Reusable Top-K Selection Kernel with Guaranteed Determinism
//
// Invariant:
// 1. Primary order: Score descending (higher score is better).
// 2. Secondary order (tie-break): VectorId ascending (lower ID wins on score equality).
//
// Computational complexity:
// - Full sort: O(N log N)
// - Bounded Quickselect (nth_element + sort top-k): O(N + K log K)
// - Bounded min-heap: O(N log K) for streaming insertions
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

#include "types.h"

namespace pomai::core {

struct TopKItem {
    VectorId id{0};
    float score{0.0f};
    uint32_t aux{0};
    const void* ptr{nullptr};
};

/// Strict weak ordering for descending score, ascending VectorId.
struct TopKScoreDescIdAsc {
    bool operator()(const TopKItem& a, const TopKItem& b) const noexcept {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.id < b.id;
    }
};

/// Min-heap ordering where the worst candidate is at top (to be popped).
/// Worst candidate = lower score, or higher VectorId on score tie.
struct TopKMinHeapComp {
    bool operator()(const TopKItem& a, const TopKItem& b) const noexcept {
        if (a.score != b.score) {
            return a.score > b.score; // min score at top
        }
        return a.id < b.id; // larger ID at top
    }
};

/**
 * SelectTopK: In-place optimal selection of the top-k elements.
 * Complexity: O(N + K log K).
 * Result items[0..k-1] are sorted according to TopKScoreDescIdAsc.
 */
inline void SelectTopK(std::vector<TopKItem>& items, size_t k) {
    if (k == 0 || items.empty()) {
        items.clear();
        return;
    }
    if (items.size() <= k) {
        std::sort(items.begin(), items.end(), TopKScoreDescIdAsc{});
        return;
    }

    // O(N) quickselect puts the k-th element at items.begin() + k
    std::nth_element(items.begin(), items.begin() + k, items.end(), TopKScoreDescIdAsc{});
    // Truncate non-top-k candidates
    items.resize(k);
    // Sort only the remaining k elements: O(K log K)
    std::sort(items.begin(), items.end(), TopKScoreDescIdAsc{});
}

/**
 * BoundedTopKQueue: Fixed-capacity heap for streaming candidate generation.
 * Size is bounded to capacity. Evicts worst candidates in O(log K).
 */
class BoundedTopKQueue {
public:
    explicit BoundedTopKQueue(size_t capacity) : capacity_(capacity) {
        heap_.reserve(capacity + 1);
    }

    [[nodiscard]] size_t size() const noexcept { return heap_.size(); }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return heap_.empty(); }

    void Push(const TopKItem& item) {
        if (capacity_ == 0) return;

        if (heap_.size() < capacity_) {
            heap_.push_back(item);
            std::push_heap(heap_.begin(), heap_.end(), TopKMinHeapComp{});
        } else {
            // Check if item is strictly better than the worst element in the heap
            // top element is heap_.front()
            // Better means: higher score, or equal score with smaller ID.
            if (item.score > heap_.front().score ||
                (item.score == heap_.front().score && item.id < heap_.front().id)) {
                std::pop_heap(heap_.begin(), heap_.end(), TopKMinHeapComp{});
                heap_.back() = item;
                std::push_heap(heap_.begin(), heap_.end(), TopKMinHeapComp{});
            }
        }
    }

    /// Extracts and sorts all items descending by score, ascending by ID.
    [[nodiscard]] std::vector<TopKItem> ExtractSorted() {
        std::vector<TopKItem> result = std::move(heap_);
        std::sort(result.begin(), result.end(), TopKScoreDescIdAsc{});
        return result;
    }

    void Clear() noexcept {
        heap_.clear();
    }

private:
    size_t capacity_{0};
    std::vector<TopKItem> heap_;
};

} // namespace pomai::core
