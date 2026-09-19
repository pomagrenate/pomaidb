// include/pio/pio_types.h — Fundamental types and Slice view for pio
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace pio {

/// Non-owning view of a contiguous sequence of bytes.
class Slice {
public:
    constexpr Slice() noexcept : data_(""), size_(0) {}
    constexpr Slice(const char* d, std::size_t n) noexcept : data_(d), size_(n) {}
    Slice(const uint8_t* d, std::size_t n) noexcept
        : data_(reinterpret_cast<const char*>(d)), size_(n) {}
    Slice(const std::string& s) noexcept : data_(s.data()), size_(s.size()) {}
    constexpr Slice(const char* s) noexcept : data_(s), size_(s ? std::char_traits<char>::length(s) : 0) {}
    constexpr Slice(std::string_view sv) noexcept : data_(sv.data()), size_(sv.size()) {}

    [[nodiscard]] constexpr const char* data() const noexcept { return data_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr char operator[](std::size_t n) const noexcept {
        return data_[n];
    }

    void clear() noexcept {
        data_ = "";
        size_ = 0;
    }

    void remove_prefix(std::size_t n) noexcept {
        assert(n <= size_);
        data_ += n;
        size_ -= n;
    }

    [[nodiscard]] std::string ToString() const {
        return std::string(data_, size_);
    }

    [[nodiscard]] std::string_view ToStringView() const noexcept {
        return std::string_view(data_, size_);
    }

    [[nodiscard]] bool starts_with(Slice x) const noexcept {
        return (size_ >= x.size_) && (std::memcmp(data_, x.data_, x.size_) == 0);
    }

    [[nodiscard]] int compare(Slice b) const noexcept {
        const std::size_t min_len = (size_ < b.size_) ? size_ : b.size_;
        int r = std::memcmp(data_, b.data_, min_len);
        if (r == 0) {
            if (size_ < b.size_) return -1;
            if (size_ > b.size_) return 1;
        }
        return r;
    }

private:
    const char* data_;
    std::size_t size_;
};

inline bool operator==(Slice x, Slice y) noexcept {
    return ((x.size() == y.size()) && (std::memcmp(x.data(), y.data(), x.size()) == 0));
}

inline bool operator!=(Slice x, Slice y) noexcept {
    return !(x == y);
}

inline bool operator<(Slice x, Slice y) noexcept {
    return x.compare(y) < 0;
}

enum class Advice : uint8_t {
    Normal = 0,
    Sequential = 1,
    Random = 2,
    WillNeed = 3,
    DontNeed = 4
};

} // namespace pio
