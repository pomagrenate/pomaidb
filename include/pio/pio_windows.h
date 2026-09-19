// include/pio/pio_windows.h — Win32 file implementations for Windows
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include "pio_file.h"
#include <vector>

#if defined(PIO_PLATFORM_WINDOWS)

namespace pio {

namespace detail {

inline std::wstring Utf8ToUtf16(const std::string& str) {
    if (str.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), NULL, 0);
    if (len <= 0) return std::wstring();
    std::wstring out(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &out[0], len);
    return out;
}

} // namespace detail

class WindowsSequentialFile final : public SequentialFile {
public:
    explicit WindowsSequentialFile(HANDLE h_file) : h_file_(h_file) {}

    ~WindowsSequentialFile() override {
        (void)Close();
    }

    Status Read(std::size_t n, Slice* result, char* scratch) override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        DWORD bytes_read = 0;
        if (!ReadFile(h_file_, scratch, static_cast<DWORD>(n), &bytes_read, NULL)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) {
                *result = Slice(scratch, 0);
                return Status::Ok();
            }
            return Status::IOError("Windows ReadFile failed with code " + std::to_string(err));
        }
        *result = Slice(scratch, static_cast<std::size_t>(bytes_read));
        return Status::Ok();
    }

    Status Skip(std::uint64_t n) override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(n);
        if (!SetFilePointerEx(h_file_, li, NULL, FILE_CURRENT)) {
            return Status::IOError("Windows SetFilePointerEx failed");
        }
        return Status::Ok();
    }

    Status Close() override {
        if (h_file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(h_file_);
            h_file_ = INVALID_HANDLE_VALUE;
        }
        return Status::Ok();
    }

private:
    HANDLE h_file_{INVALID_HANDLE_VALUE};
};

class WindowsRandomAccessFile final : public RandomAccessFile {
public:
    explicit WindowsRandomAccessFile(HANDLE h_file) : h_file_(h_file) {}

    ~WindowsRandomAccessFile() override {
        (void)Close();
    }

    Status Read(std::uint64_t offset, std::size_t n, Slice* result, char* scratch) const override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
        DWORD bytes_read = 0;
        if (!ReadFile(h_file_, scratch, static_cast<DWORD>(n), &bytes_read, &ov)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) {
                *result = Slice(scratch, 0);
                return Status::Ok();
            }
            return Status::IOError("Windows ReadFile failed with code " + std::to_string(err));
        }
        *result = Slice(scratch, static_cast<std::size_t>(bytes_read));
        return Status::Ok();
    }

    void Prefetch(std::uint64_t /*offset*/, std::size_t /*n*/) const override {}

    Status Close() override {
        if (h_file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(h_file_);
            h_file_ = INVALID_HANDLE_VALUE;
        }
        return Status::Ok();
    }

private:
    mutable HANDLE h_file_{INVALID_HANDLE_VALUE};
};

class WindowsWritableFile final : public WritableFile {
public:
    explicit WindowsWritableFile(HANDLE h_file, std::size_t buffer_size = kDefaultWriteBufferSize)
        : h_file_(h_file), buffer_capacity_(buffer_size) {
        buf_.resize(buffer_capacity_);
    }

    ~WindowsWritableFile() override {
        (void)Close();
    }

    Status Append(Slice data) override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        const char* src = data.data();
        std::size_t size = data.size();

        if (buf_used_ + size <= buffer_capacity_) {
            std::memcpy(buf_.data() + buf_used_, src, size);
            buf_used_ += size;
            return Status::Ok();
        }

        if (buf_used_ > 0) {
            Status s = Flush();
            if (!s.ok()) return s;
        }

        if (size >= buffer_capacity_ / 2) {
            return WriteDirect(src, size);
        }

        std::memcpy(buf_.data(), src, size);
        buf_used_ = size;
        return Status::Ok();
    }

    Status Pwrite(std::uint64_t offset, Slice data) override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        if (buf_used_ > 0) {
            Status s = Flush();
            if (!s.ok()) return s;
        }
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
        DWORD written = 0;
        if (!WriteFile(h_file_, data.data(), static_cast<DWORD>(data.size()), &written, &ov)) {
            return Status::IOError("Windows WriteFile positional write failed");
        }
        return Status::Ok();
    }

    Status Flush() override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        if (buf_used_ == 0) return Status::Ok();
        Status s = WriteDirect(buf_.data(), buf_used_);
        buf_used_ = 0;
        return s;
    }

    Status Sync() override {
        Status s = Flush();
        if (!s.ok()) return s;
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::IOError("file is closed");
        if (!FlushFileBuffers(h_file_)) {
            return Status::IOError("Windows FlushFileBuffers failed");
        }
        return Status::Ok();
    }

    Status Close() override {
        if (h_file_ == INVALID_HANDLE_VALUE) return Status::Ok();
        Status s = Flush();
        CloseHandle(h_file_);
        h_file_ = INVALID_HANDLE_VALUE;
        return s;
    }

    [[nodiscard]] std::uint64_t BytesWritten() const noexcept override {
        return bytes_written_ + buf_used_;
    }

private:
    Status WriteDirect(const char* data, std::size_t size) {
        const char* p = data;
        std::size_t rem = size;
        while (rem > 0) {
            DWORD to_write = static_cast<DWORD>(std::min<std::size_t>(rem, 64 * 1024 * 1024));
            DWORD written = 0;
            if (!WriteFile(h_file_, p, to_write, &written, NULL)) {
                return Status::IOError("Windows WriteFile failed");
            }
            p += written;
            rem -= written;
            bytes_written_ += written;
        }
        return Status::Ok();
    }

    HANDLE h_file_{INVALID_HANDLE_VALUE};
    std::size_t buffer_capacity_{kDefaultWriteBufferSize};
    std::size_t buf_used_{0};
    std::uint64_t bytes_written_{0};
    std::vector<char> buf_;
};

class WindowsMemoryMappedFile final : public MemoryMappedFile {
public:
    WindowsMemoryMappedFile(HANDLE h_file, HANDLE h_map, const std::uint8_t* data, std::size_t size)
        : h_file_(h_file), h_map_(h_map), data_(data), size_(size) {}

    ~WindowsMemoryMappedFile() override {
        Close();
    }

    [[nodiscard]] const std::uint8_t* data() const noexcept override { return data_; }
    [[nodiscard]] std::size_t size() const noexcept override { return size_; }

    void Advise(Advice /*advice*/) override {}

    void Close() override {
        if (data_) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
            size_ = 0;
        }
        if (h_map_) {
            CloseHandle(h_map_);
            h_map_ = NULL;
        }
        if (h_file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(h_file_);
            h_file_ = INVALID_HANDLE_VALUE;
        }
    }

private:
    HANDLE h_file_{INVALID_HANDLE_VALUE};
    HANDLE h_map_{NULL};
    const std::uint8_t* data_{nullptr};
    std::size_t size_{0};
};

} // namespace pio

#endif // PIO_PLATFORM_WINDOWS
