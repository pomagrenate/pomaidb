// include/pio/pio_posix.h — POSIX file implementations for Linux and macOS
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include "pio_file.h"
#include <vector>

#if defined(PIO_PLATFORM_POSIX)

namespace pio {

class PosixSequentialFile final : public SequentialFile {
public:
    explicit PosixSequentialFile(int fd) : fd_(fd) {
        if (fd_ >= 0) {
#if defined(__linux__)
            ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
        }
    }

    ~PosixSequentialFile() override {
        (void)Close();
    }

    Status Read(std::size_t n, Slice* result, char* scratch) override {
        if (fd_ < 0) return Status::IOError("file is closed");
        ssize_t r = ::read(fd_, scratch, n);
        if (r < 0) {
            if (errno == EINTR) return Read(n, result, scratch);
            return Status::IOError("Posix read failed: " + std::string(strerror(errno)));
        }
        *result = Slice(scratch, static_cast<std::size_t>(r));
        return Status::Ok();
    }

    Status Skip(std::uint64_t n) override {
        if (fd_ < 0) return Status::IOError("file is closed");
        if (::lseek(fd_, static_cast<off_t>(n), SEEK_CUR) == -1) {
            return Status::IOError("Posix lseek failed: " + std::string(strerror(errno)));
        }
        return Status::Ok();
    }

    Status Close() override {
        if (fd_ >= 0) {
            int res = ::close(fd_);
            fd_ = -1;
            if (res != 0) return Status::IOError("Posix close failed");
        }
        return Status::Ok();
    }

private:
    int fd_{-1};
};

class PosixRandomAccessFile final : public RandomAccessFile {
public:
    explicit PosixRandomAccessFile(int fd) : fd_(fd) {}

    ~PosixRandomAccessFile() override {
        (void)Close();
    }

    Status Read(std::uint64_t offset, std::size_t n, Slice* result, char* scratch) const override {
        if (fd_ < 0) return Status::IOError("file is closed");
        ssize_t r = ::pread(fd_, scratch, n, static_cast<off_t>(offset));
        if (r < 0) {
            if (errno == EINTR) return Read(offset, n, result, scratch);
            return Status::IOError("Posix pread failed: " + std::string(strerror(errno)));
        }
        *result = Slice(scratch, static_cast<std::size_t>(r));
        return Status::Ok();
    }

    void Prefetch(std::uint64_t offset, std::size_t n) const override {
        if (fd_ >= 0) {
#if defined(__linux__)
            ::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(n), POSIX_FADV_WILLNEED);
#endif
        }
    }

    Status Close() override {
        if (fd_ >= 0) {
            int res = ::close(fd_);
            fd_ = -1;
            if (res != 0) return Status::IOError("Posix close failed");
        }
        return Status::Ok();
    }

private:
    mutable int fd_{-1};
};

class PosixWritableFile final : public WritableFile {
public:
    explicit PosixWritableFile(int fd, std::size_t buffer_size = kDefaultWriteBufferSize)
        : fd_(fd), buffer_capacity_(buffer_size) {
        buf_.resize(buffer_capacity_);
    }

    ~PosixWritableFile() override {
        (void)Close();
    }

    Status Append(Slice data) override {
        if (fd_ < 0) return Status::IOError("file is closed");
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
        if (fd_ < 0) return Status::IOError("file is closed");
        if (buf_used_ > 0) {
            Status s = Flush();
            if (!s.ok()) return s;
        }
        ssize_t w = ::pwrite(fd_, data.data(), data.size(), static_cast<off_t>(offset));
        if (w < 0) return Status::IOError("Posix pwrite failed");
        return Status::Ok();
    }

    Status Flush() override {
        if (fd_ < 0) return Status::IOError("file is closed");
        if (buf_used_ == 0) return Status::Ok();
        Status s = WriteDirect(buf_.data(), buf_used_);
        buf_used_ = 0;
        return s;
    }

    Status Sync() override {
        Status s = Flush();
        if (!s.ok()) return s;
        if (fd_ < 0) return Status::IOError("file is closed");
#if defined(__APPLE__)
        return ::fsync(fd_) == 0 ? Status::Ok() : Status::IOError("fsync failed");
#else
        return ::fdatasync(fd_) == 0 ? Status::Ok() : Status::IOError("fdatasync failed");
#endif
    }

    Status Close() override {
        if (fd_ < 0) return Status::Ok();
        Status s = Flush();
        int res = ::close(fd_);
        fd_ = -1;
        if (!s.ok()) return s;
        return res == 0 ? Status::Ok() : Status::IOError("close failed");
    }

    [[nodiscard]] std::uint64_t BytesWritten() const noexcept override {
        return bytes_written_ + buf_used_;
    }

private:
    Status WriteDirect(const char* data, std::size_t size) {
        const char* p = data;
        std::size_t rem = size;
        while (rem > 0) {
            ssize_t w = ::write(fd_, p, rem);
            if (w < 0) {
                if (errno == EINTR) continue;
                return Status::IOError("Posix write failed: " + std::string(strerror(errno)));
            }
            p += static_cast<std::size_t>(w);
            rem -= static_cast<std::size_t>(w);
            bytes_written_ += static_cast<std::uint64_t>(w);
        }
        return Status::Ok();
    }

    int fd_{-1};
    std::size_t buffer_capacity_{kDefaultWriteBufferSize};
    std::size_t buf_used_{0};
    std::uint64_t bytes_written_{0};
    std::vector<char> buf_;
};

class PosixMemoryMappedFile final : public MemoryMappedFile {
public:
    PosixMemoryMappedFile(int fd, const std::uint8_t* data, std::size_t size)
        : fd_(fd), data_(data), size_(size) {}

    ~PosixMemoryMappedFile() override {
        Close();
    }

    [[nodiscard]] const std::uint8_t* data() const noexcept override { return data_; }
    [[nodiscard]] std::size_t size() const noexcept override { return size_; }

    void Advise(Advice advice) override {
        if (!data_ || size_ == 0) return;
        int adv = MADV_NORMAL;
        switch (advice) {
            case Advice::Sequential: adv = MADV_SEQUENTIAL; break;
            case Advice::Random: adv = MADV_RANDOM; break;
            case Advice::WillNeed: adv = MADV_WILLNEED; break;
            case Advice::DontNeed: adv = MADV_DONTNEED; break;
            default: adv = MADV_NORMAL; break;
        }
        (void)::madvise(const_cast<void*>(reinterpret_cast<const void*>(data_)), size_, adv);
    }

    void Close() override {
        if (data_ && size_ > 0) {
            ::munmap(const_cast<uint8_t*>(data_), size_);
            data_ = nullptr;
            size_ = 0;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_{-1};
    const std::uint8_t* data_{nullptr};
    std::size_t size_{0};
};

} // namespace pio

#endif // PIO_PLATFORM_POSIX
