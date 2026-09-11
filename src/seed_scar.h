// pomai/seed_scar.h — Tombstone bitset representing deleted vector states (Seed Scar)
//
// In PomaiDB, deleted vectors in an immutable Aril are marked in the Seed Scar.
// Press (compaction) physically dries (removes) vectors marked in the Seed Scar.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pomai::storage {

/**
 * SeedScarView: Read-only zero-copy view of the tombstone bitset in an Aril.
 */
class SeedScarView {
public:
    constexpr SeedScarView() = default;
    constexpr SeedScarView(const uint8_t* data, uint32_t count)
        : data_(data), count_(count) {}

    [[nodiscard]] bool IsDeleted(uint32_t slot) const noexcept {
        if (!data_ || slot >= count_) return false;
        return (data_[slot >> 3] & (1u << (slot & 7))) != 0;
    }

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return (count_ + 7) >> 3; }
    [[nodiscard]] const uint8_t* data() const noexcept { return data_; }

    [[nodiscard]] uint32_t CountDeleted() const noexcept {
        if (!data_) return 0;
        uint32_t deleted = 0;
        const size_t bytes = size_bytes();
        for (size_t i = 0; i < bytes; ++i) {
            deleted += static_cast<uint32_t>(__builtin_popcount(data_[i]));
        }
        return deleted;
    }

    [[nodiscard]] uint32_t deleted_count() const noexcept {
        return CountDeleted();
    }

private:
    const uint8_t* data_{nullptr};
    uint32_t count_{0};
};

/**
 * SeedScarBuilder: Mutable bitset builder used while assembling an Aril.
 */
class SeedScarBuilder {
public:
    explicit SeedScarBuilder(uint32_t count)
        : count_(count), bytes_((count + 7) >> 3, 0) {}

    void MarkDeleted(uint32_t slot) noexcept {
        if (slot < count_) {
            bytes_[slot >> 3] |= static_cast<uint8_t>(1u << (slot & 7));
        }
    }

    [[nodiscard]] bool IsDeleted(uint32_t slot) const noexcept {
        if (slot >= count_) return false;
        return (bytes_[slot >> 3] & (1u << (slot & 7))) != 0;
    }

    [[nodiscard]] std::span<const uint8_t> bytes() const noexcept {
        return bytes_;
    }

    [[nodiscard]] uint32_t count() const noexcept { return count_; }

private:
    uint32_t count_{0};
    std::vector<uint8_t> bytes_;
};

} // namespace pomai::storage
