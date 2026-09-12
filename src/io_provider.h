#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "status.h"
#include "types.h"

namespace pomai::storage {

/** Chunk size for streaming file reads (1MB) to bound memory on embedded. */
constexpr size_t kStreamReadChunkSize = 1024 * 1024;

/**
 * SequentialFile: Optimized for streaming reads (e.g. WAL replay, Scan).
 */
class SequentialFile {
 public:
  virtual ~SequentialFile() = default;
  virtual Status Read(size_t n, Slice* result, char* scratch) = 0;
  virtual Status Skip(uint64_t n) = 0;
};

/**
 * RandomAccessFile: Optimized for point lookups (e.g. Vector Search).
 */
class RandomAccessFile {
 public:
  virtual ~RandomAccessFile() = default;
  virtual Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const = 0;
  
  // Hint that data will be needed soon (POSIX_FADV_WILLNEED)
  virtual void Prefetch(uint64_t offset, size_t n) const = 0;
};

/**
 * WritableFile: Optimized for append-only writing with explicit sync gates.
 */
class WritableFile {
 public:
  virtual ~WritableFile() = default;
  virtual Status Append(Slice data) = 0;
  virtual Status Pwrite(uint64_t offset, Slice data) = 0;
  virtual uint64_t BytesWritten() const = 0;
  virtual Status Flush() = 0;
  virtual Status Sync() = 0;
  virtual Status Close() = 0;
};

/**
 * MemoryMappedFile: Interface for zero-copy file access.
 */
class MemoryMappedFile {
 public:
  virtual ~MemoryMappedFile() = default;
  virtual const uint8_t* Data() const = 0;
  virtual size_t Size() const = 0;
};

#if !defined(_WIN32)
class PosixMemoryMappedFile : public MemoryMappedFile {
 public:
  PosixMemoryMappedFile(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  ~PosixMemoryMappedFile() override {
    if (data_) munmap(const_cast<uint8_t*>(data_), size_);
  }
  const uint8_t* Data() const override { return data_; }
  size_t Size() const override { return size_; }
 private:
  const uint8_t* data_;
  size_t size_;
};

class PosixSequentialFile : public SequentialFile {
 public:
  explicit PosixSequentialFile(int fd) : fd_(fd) {
    ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
  }
  ~PosixSequentialFile() override { ::close(fd_); }

  Status Read(size_t n, Slice* result, char* scratch) override {
    ssize_t r = ::read(fd_, scratch, n);
    if (r < 0) return Status::IOError("Sequential Read failed");
    *result = Slice(scratch, static_cast<size_t>(r));
    return Status::Ok();
  }

  Status Skip(uint64_t n) override {
    if (::lseek(fd_, static_cast<off_t>(n), SEEK_CUR) == -1) {
      return Status::IOError("Skip failed");
    }
    return Status::Ok();
  }

 private:
  int fd_;
};

class PosixRandomAccessFile : public RandomAccessFile {
 public:
  explicit PosixRandomAccessFile(int fd) : fd_(fd) {}
  ~PosixRandomAccessFile() override { ::close(fd_); }

  Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const override {
    ssize_t r = ::pread(fd_, scratch, n, static_cast<off_t>(offset));
    if (r < 0) return Status::IOError("Random Read failed");
    *result = Slice(scratch, static_cast<size_t>(r));
    return Status::Ok();
  }

  void Prefetch(uint64_t offset, size_t n) const override {
    ::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(n), POSIX_FADV_WILLNEED);
  }

 private:
  int fd_;
};

class PosixWritableFile : public WritableFile {
 public:
  static constexpr size_t kBufferSize = 65536;

  explicit PosixWritableFile(int fd)
      : fd_(fd), bytes_written_(0), buf_used_(0) {
    buf_.resize(kBufferSize);
  }
  ~PosixWritableFile() override { (void)Close(); }

  Status Append(Slice data) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    const char* src = reinterpret_cast<const char*>(data.data());
    size_t size = data.size();

    if (buf_used_ + size <= kBufferSize) {
      std::memcpy(buf_.data() + buf_used_, src, size);
      buf_used_ += size;
      return Status::Ok();
    }

    if (buf_used_ > 0) {
      Status s = FlushBuffer();
      if (!s.ok()) return s;
    }

    if (size >= kBufferSize / 2) {
      return WriteDirect(src, size);
    }

    std::memcpy(buf_.data(), src, size);
    buf_used_ = size;
    return Status::Ok();
  }

  Status Pwrite(uint64_t offset, Slice data) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    if (buf_used_ > 0) {
      Status s = FlushBuffer();
      if (!s.ok()) return s;
    }
    ssize_t w = ::pwrite(fd_, data.data(), data.size(), static_cast<off_t>(offset));
    if (w < 0) return Status::IOError("Pwrite failed");
    return Status::Ok();
  }

  uint64_t BytesWritten() const override { return bytes_written_ + buf_used_; }

  Status Flush() override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    return FlushBuffer();
  }

  Status Sync() override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    Status s = FlushBuffer();
    if (!s.ok()) return s;
    return ::fdatasync(fd_) == 0 ? Status::Ok() : Status::IOError("Sync failed");
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    Status s = FlushBuffer();
    int res = ::close(fd_);
    fd_ = -1;
    if (!s.ok()) return s;
    return res == 0 ? Status::Ok() : Status::IOError("Close failed");
  }

 private:
  Status WriteDirect(const char* data, size_t size) {
    const char* p = data;
    size_t rem = size;
    while (rem > 0) {
      ssize_t w = ::write(fd_, p, rem);
      if (w < 0) {
        if (errno == EINTR) continue;
        return Status::IOError("Append failed");
      }
      p += static_cast<size_t>(w);
      rem -= static_cast<size_t>(w);
      bytes_written_ += static_cast<uint64_t>(w);
    }
    return Status::Ok();
  }

  Status FlushBuffer() {
    if (buf_used_ == 0) return Status::Ok();
    Status s = WriteDirect(buf_.data(), buf_used_);
    buf_used_ = 0;
    return s;
  }

 protected:
  int fd_;
  uint64_t bytes_written_ = 0;
  size_t buf_used_ = 0;
  std::vector<char> buf_;
};

#else // _WIN32

class WindowsMemoryMappedFile : public MemoryMappedFile {
 public:
  WindowsMemoryMappedFile(HANDLE h_file, HANDLE h_map, const uint8_t* data, size_t size)
      : h_file_(h_file), h_map_(h_map), data_(data), size_(size) {}
  ~WindowsMemoryMappedFile() override {
    if (data_) UnmapViewOfFile(data_);
    if (h_map_) CloseHandle(h_map_);
    if (h_file_ != INVALID_HANDLE_VALUE) CloseHandle(h_file_);
  }
  const uint8_t* Data() const override { return data_; }
  size_t Size() const override { return size_; }
 private:
  HANDLE h_file_ = INVALID_HANDLE_VALUE;
  HANDLE h_map_ = NULL;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

class WindowsSequentialFile : public SequentialFile {
 public:
  explicit WindowsSequentialFile(HANDLE h_file) : h_file_(h_file) {}
  ~WindowsSequentialFile() override {
    if (h_file_ != INVALID_HANDLE_VALUE) CloseHandle(h_file_);
  }
  Status Read(size_t n, Slice* result, char* scratch) override {
    DWORD bytes_read = 0;
    if (!ReadFile(h_file_, scratch, static_cast<DWORD>(n), &bytes_read, NULL)) {
      return Status::IOError("Sequential Read failed");
    }
    *result = Slice(scratch, static_cast<size_t>(bytes_read));
    return Status::Ok();
  }
  Status Skip(uint64_t n) override {
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(n);
    if (!SetFilePointerEx(h_file_, li, NULL, FILE_CURRENT)) {
      return Status::IOError("Skip failed");
    }
    return Status::Ok();
  }
 private:
  HANDLE h_file_ = INVALID_HANDLE_VALUE;
};

class WindowsRandomAccessFile : public RandomAccessFile {
 public:
  explicit WindowsRandomAccessFile(HANDLE h_file) : h_file_(h_file) {}
  ~WindowsRandomAccessFile() override {
    if (h_file_ != INVALID_HANDLE_VALUE) CloseHandle(h_file_);
  }
  Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const override {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
    DWORD bytes_read = 0;
    if (!ReadFile(h_file_, scratch, static_cast<DWORD>(n), &bytes_read, &ov)) {
      return Status::IOError("Random Read failed");
    }
    *result = Slice(scratch, static_cast<size_t>(bytes_read));
    return Status::Ok();
  }
  void Prefetch(uint64_t /*offset*/, size_t /*n*/) const override {}
 private:
  HANDLE h_file_ = INVALID_HANDLE_VALUE;
};

class WindowsWritableFile : public WritableFile {
 public:
  static constexpr size_t kBufferSize = 65536;

  explicit WindowsWritableFile(HANDLE h_file)
      : h_file_(h_file), bytes_written_(0), buf_used_(0) {
    buf_.resize(kBufferSize);
  }
  ~WindowsWritableFile() override {
    (void)Close();
  }

  Status Append(Slice data) override {
    if (h_file_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    const char* src = reinterpret_cast<const char*>(data.data());
    size_t size = data.size();

    if (buf_used_ + size <= kBufferSize) {
      std::memcpy(buf_.data() + buf_used_, src, size);
      buf_used_ += size;
      return Status::Ok();
    }

    if (buf_used_ > 0) {
      Status s = FlushBuffer();
      if (!s.ok()) return s;
    }

    if (size >= kBufferSize / 2) {
      return WriteDirect(src, size);
    }

    std::memcpy(buf_.data(), src, size);
    buf_used_ = size;
    return Status::Ok();
  }

  Status Pwrite(uint64_t offset, Slice data) override {
    if (h_file_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    if (buf_used_ > 0) {
      Status s = FlushBuffer();
      if (!s.ok()) return s;
    }
    LARGE_INTEGER liOffset;
    liOffset.QuadPart = static_cast<LONGLONG>(offset);
    if (!SetFilePointerEx(h_file_, liOffset, NULL, FILE_BEGIN)) {
      return Status::IOError("Pwrite seek failed");
    }
    DWORD written = 0;
    if (!WriteFile(h_file_, data.data(), static_cast<DWORD>(data.size()), &written, NULL)) {
      return Status::IOError("Pwrite failed");
    }
    return Status::Ok();
  }

  uint64_t BytesWritten() const override { return bytes_written_ + buf_used_; }

  Status Flush() override {
    if (h_file_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    return FlushBuffer();
  }

  Status Sync() override {
    if (h_file_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    Status s = FlushBuffer();
    if (!s.ok()) return s;
    return FlushFileBuffers(h_file_) ? Status::Ok() : Status::IOError("Sync failed");
  }

  Status Close() override {
    if (h_file_ != INVALID_HANDLE_VALUE) {
      Status s = FlushBuffer();
      CloseHandle(h_file_);
      h_file_ = INVALID_HANDLE_VALUE;
      return s;
    }
    return Status::Ok();
  }

 private:
  Status WriteDirect(const char* data, size_t size) {
    LARGE_INTEGER liZero{};
    LARGE_INTEGER liEnd{};
    if (!SetFilePointerEx(h_file_, liZero, &liEnd, FILE_END)) {
      return Status::IOError("Append seek failed");
    }
    DWORD written = 0;
    if (!WriteFile(h_file_, data, static_cast<DWORD>(size), &written, NULL) || written != size) {
      return Status::IOError("Append failed");
    }
    bytes_written_ = static_cast<uint64_t>(liEnd.QuadPart) + written;
    return Status::Ok();
  }

  Status FlushBuffer() {
    if (buf_used_ == 0) return Status::Ok();
    Status s = WriteDirect(buf_.data(), buf_used_);
    buf_used_ = 0;
    return s;
  }

 protected:
  HANDLE h_file_ = INVALID_HANDLE_VALUE;
  uint64_t bytes_written_ = 0;
  size_t buf_used_ = 0;
  std::vector<char> buf_;
};

#endif

class PosixIOProvider {
 public:
  static Status NewSequentialFile(const std::filesystem::path& path, std::unique_ptr<SequentialFile>* result);
  static Status NewRandomAccessFile(const std::filesystem::path& path, std::unique_ptr<RandomAccessFile>* result);
  static Status NewWritableFile(const std::filesystem::path& path, std::unique_ptr<WritableFile>* result);
  static Status NewMemoryMappedFile(const std::filesystem::path& path, std::unique_ptr<MemoryMappedFile>* result);
};

// --- Factory Implementations ---

inline Status PosixIOProvider::NewSequentialFile(const std::filesystem::path& path, std::unique_ptr<SequentialFile>* result) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<WindowsSequentialFile>(h);
    return Status::Ok();
#else
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<PosixSequentialFile>(fd);
    return Status::Ok();
#endif
}

inline Status PosixIOProvider::NewRandomAccessFile(const std::filesystem::path& path, std::unique_ptr<RandomAccessFile>* result) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<WindowsRandomAccessFile>(h);
    return Status::Ok();
#else
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<PosixRandomAccessFile>(fd);
    return Status::Ok();
#endif
}

inline Status PosixIOProvider::NewWritableFile(const std::filesystem::path& path, std::unique_ptr<WritableFile>* result) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<WindowsWritableFile>(h);
    return Status::Ok();
#else
    int fd = ::open(path.c_str(), O_TRUNC | O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<PosixWritableFile>(fd);
    return Status::Ok();
#endif
}

inline Status PosixIOProvider::NewMemoryMappedFile(const std::filesystem::path& path, std::unique_ptr<MemoryMappedFile>* result) {
#if defined(_WIN32)
    HANDLE h_file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h_file == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path.string());
    LARGE_INTEGER li;
    if (!GetFileSizeEx(h_file, &li)) {
      CloseHandle(h_file);
      return Status::IOError("GetFileSizeEx failed: " + path.string());
    }
    size_t size = static_cast<size_t>(li.QuadPart);
    if (size == 0) {
      *result = std::make_unique<WindowsMemoryMappedFile>(h_file, static_cast<HANDLE>(NULL), nullptr, 0);
      return Status::Ok();
    }
    HANDLE h_map = CreateFileMappingW(h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!h_map) {
      CloseHandle(h_file);
      return Status::IOError("CreateFileMapping failed: " + path.string());
    }
    void* data = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, size);
    if (!data) {
      CloseHandle(h_map);
      CloseHandle(h_file);
      return Status::IOError("MapViewOfFile failed: " + path.string());
    }
    *result = std::make_unique<WindowsMemoryMappedFile>(h_file, h_map, static_cast<const uint8_t*>(data), size);
    return Status::Ok();
#else
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      return Status::IOError("fstat failed: " + path.string());
    }
    size_t size = st.st_size;
    void* data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (data == MAP_FAILED) return Status::IOError("mmap failed: " + path.string());
    *result = std::make_unique<PosixMemoryMappedFile>(static_cast<const uint8_t*>(data), size);
    return Status::Ok();
#endif
}

} // namespace pomai::storage
