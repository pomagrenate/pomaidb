// include/pio/pio_status.h — Lightweight error and status representation for pio
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include <string>
#include <string_view>
#include "pio_types.h"

namespace pio {

class [[nodiscard]] Status {
public:
    enum Code : uint8_t {
        kOk = 0,
        kNotFound = 1,
        kCorruption = 2,
        kNotSupported = 3,
        kInvalidArgument = 4,
        kIOError = 5,
        kAlreadyExists = 6
    };

    constexpr Status() noexcept : code_(kOk), msg_("") {}
    explicit Status(Code c, std::string msg = "") : code_(c), msg_(std::move(msg)) {}

    static Status Ok() noexcept { return Status(); }
    static Status NotFound(std::string msg = "") { return Status(kNotFound, std::move(msg)); }
    static Status Corruption(std::string msg = "") { return Status(kCorruption, std::move(msg)); }
    static Status NotSupported(std::string msg = "") { return Status(kNotSupported, std::move(msg)); }
    static Status InvalidArgument(std::string msg = "") { return Status(kInvalidArgument, std::move(msg)); }
    static Status IOError(std::string msg = "") { return Status(kIOError, std::move(msg)); }
    static Status AlreadyExists(std::string msg = "") { return Status(kAlreadyExists, std::move(msg)); }

    [[nodiscard]] constexpr bool ok() const noexcept { return code_ == kOk; }
    [[nodiscard]] constexpr bool IsNotFound() const noexcept { return code_ == kNotFound; }
    [[nodiscard]] constexpr bool IsCorruption() const noexcept { return code_ == kCorruption; }
    [[nodiscard]] constexpr bool IsNotSupported() const noexcept { return code_ == kNotSupported; }
    [[nodiscard]] constexpr bool IsInvalidArgument() const noexcept { return code_ == kInvalidArgument; }
    [[nodiscard]] constexpr bool IsIOError() const noexcept { return code_ == kIOError; }
    [[nodiscard]] constexpr bool IsAlreadyExists() const noexcept { return code_ == kAlreadyExists; }
    [[nodiscard]] constexpr Code code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return msg_; }

    [[nodiscard]] std::string ToString() const {
        if (code_ == kOk) return "OK";
        std::string s;
        switch (code_) {
            case kNotFound: s = "NotFound: "; break;
            case kCorruption: s = "Corruption: "; break;
            case kNotSupported: s = "NotSupported: "; break;
            case kInvalidArgument: s = "InvalidArgument: "; break;
            case kIOError: s = "IOError: "; break;
            case kAlreadyExists: s = "AlreadyExists: "; break;
            default: s = "Unknown: "; break;
        }
        s.append(msg_);
        return s;
    }

private:
    Code code_{kOk};
    std::string msg_;
};

} // namespace pio
