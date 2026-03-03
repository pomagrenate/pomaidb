// router.h — Consistent-hash MembraneID→ShardID router for PomaiDB.
// Single-threaded event-loop: no mutex, no atomics.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace pomai::core {

// ── Consistent-hash helper ───────────────────────────────────────────────────
// Maps an arbitrary 64-bit key (MembraneID or VectorId) to a shard index in
// [0, num_shards) using jump consistent hashing (Google 2014, 3 lines).
// Deterministic, zero-allocation, O(log n) work.
inline uint32_t JumpConsistentHash(uint64_t key, uint32_t num_shards) noexcept {
    int64_t b = -1, j = 0;
    while (j < static_cast<int64_t>(num_shards)) {
        b = j;
        key = key * 2862933555777941757ULL + 1;
        j = static_cast<int64_t>(
            static_cast<double>(b + 1) *
            (static_cast<double>(1LL << 31) /
             static_cast<double>((key >> 33) + 1)));
    }
    return static_cast<uint32_t>(b);
}

// ── ShardRouter — sequential routing table ───────────────────────────────────
class ShardRouter {
public:
    explicit ShardRouter(uint32_t num_shards) noexcept
        : num_shards_(num_shards) {}

    uint32_t RouteByKey(uint64_t key) const noexcept {
        return JumpConsistentHash(key, num_shards_);
    }

    uint32_t RouteByVector(uint64_t key, std::span<const float> /*vec*/) const noexcept {
        return RouteByKey(key);
    }

    uint32_t num_shards() const noexcept { return num_shards_; }

    template <typename Fn>
    void Update(Fn&& fn) {
        fn();
    }

private:
    const uint32_t num_shards_;
};

} // namespace pomai::core
