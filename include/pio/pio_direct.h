// include/pio/pio_direct.h — Sector-aligned Direct I/O for NVMe and flash storage
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "pio_platform.h"
#include "pio_status.h"
#include "pio_types.h"

namespace pio {

/// Sector-aligned Direct I/O buffer helper.
/// Guaranteed 4096-byte alignment to bypass OS kernel page cache (O_DIRECT / NO_BUFFERING).
class DirectBuffer {
public:
    explicit DirectBuffer(std::size_t capacity)
        : capacity_((capacity + kDirectIoSectorSize - 1) & ~(kDirectIoSectorSize - 1)) {
        data_ = static_cast<uint8_t*>(direct_alloc(capacity_, kDirectIoSectorSize));
    }

    ~DirectBuffer() {
        if (data_) {
            direct_free(data_);
            data_ = nullptr;
        }
    }

    DirectBuffer(const DirectBuffer&) = delete;
    DirectBuffer& operator=(const DirectBuffer&) = delete;

    DirectBuffer(DirectBuffer&& o) noexcept
        : data_(o.data_), capacity_(o.capacity_), size_(o.size_) {
        o.data_ = nullptr;
        o.capacity_ = 0;
        o.size_ = 0;
    }

    DirectBuffer& operator=(DirectBuffer&& o) noexcept {
        if (this != &o) {
            if (data_) direct_free(data_);
            data_ = o.data_;
            capacity_ = o.capacity_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.capacity_ = 0;
            o.size_ = 0;
        }
        return *this;
    }

    [[nodiscard]] uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] bool IsAligned() const noexcept {
        return (reinterpret_cast<uintptr_t>(data_) % kDirectIoSectorSize) == 0;
    }

    void set_size(std::size_t s) noexcept { size_ = s; }
    void clear() noexcept { size_ = 0; }

    void resize(std::size_t new_cap) {
        std::size_t aligned_cap = (new_cap + kDirectIoSectorSize - 1) & ~(kDirectIoSectorSize - 1);
        if (aligned_cap == capacity_) return;
        uint8_t* new_data = static_cast<uint8_t*>(direct_alloc(aligned_cap, kDirectIoSectorSize));
        if (data_) {
            std::size_t copy_len = std::min(size_, aligned_cap);
            std::memcpy(new_data, data_, copy_len);
            direct_free(data_);
        }
        data_ = new_data;
        capacity_ = aligned_cap;
        size_ = std::min(size_, aligned_cap);
    }

private:
    uint8_t* data_{nullptr};
    std::size_t capacity_{0};
    std::size_t size_{0};
};

using AlignedBuffer = DirectBuffer;

} // namespace pio
