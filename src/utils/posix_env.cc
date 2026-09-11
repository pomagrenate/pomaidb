// util/posix_env.cc — POSIX-backed Env implementation.
// All OS-specific includes and calls are confined to this file.

#include "posix_env.h"

#include <cerrno>
#include <cstring>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <filesystem>
#ifdef DeleteFile
#undef DeleteFile
#endif
#ifdef CreateFile
#undef CreateFile
#endif
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace pomai {

#if !defined(_WIN32) && !defined(_WIN64)

namespace detail {

static Status ErrnoStatus(const char* op) {
  return Status::IOError(std::string(op) + ": " + std::strerror(errno));
}

// -----------------------------------------------------------------------------
// PosixSequentialFile
// -----------------------------------------------------------------------------
class PosixSequentialFile : public SequentialFile {
 public:
  explicit PosixSequentialFile(int fd) : fd_(fd) {}
  ~PosixSequentialFile() override { (void)Close(); }

  Status Read(size_t n, Slice* result) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    buf_.resize(n);
    ssize_t r = ::read(fd_, buf_.data(), n);
    if (r < 0) {
      if (errno == EINTR) return Read(n, result);
      return ErrnoStatus("read");
    }
    buf_.resize(static_cast<size_t>(r));
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Skip(uint64_t n) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    if (::lseek(fd_, static_cast<off_t>(n), SEEK_CUR) == static_cast<off_t>(-1))
      return ErrnoStatus("lseek");
    return Status::Ok();
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    int r;
    do {
      r = ::close(fd_);
    } while (r != 0 && errno == EINTR);
    fd_ = -1;
    return r == 0 ? Status::Ok() : ErrnoStatus("close");
  }

 private:
  int fd_ = -1;
  std::vector<char> buf_;
};

// -----------------------------------------------------------------------------
// PosixRandomAccessFile
// -----------------------------------------------------------------------------
class PosixRandomAccessFile : public RandomAccessFile {
 public:
  explicit PosixRandomAccessFile(int fd) : fd_(fd) {}
  ~PosixRandomAccessFile() override { (void)Close(); }

  Status Read(uint64_t offset, size_t n, Slice* result) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    buf_.resize(n);
    ssize_t r = ::pread(fd_, buf_.data(), n, static_cast<off_t>(offset));
    if (r < 0) {
      if (errno == EINTR) return Read(offset, n, result);
      return ErrnoStatus("pread");
    }
    buf_.resize(static_cast<size_t>(r));
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    int r;
    do {
      r = ::close(fd_);
    } while (r != 0 && errno == EINTR);
    fd_ = -1;
    return r == 0 ? Status::Ok() : ErrnoStatus("close");
  }

 private:
  int fd_ = -1;
  mutable std::vector<char> buf_;
};

// -----------------------------------------------------------------------------
// PosixWritableFile
// -----------------------------------------------------------------------------
class PosixWritableFile : public WritableFile {
 public:
  static constexpr size_t kBufferSize = 65536;

  PosixWritableFile(int fd, uint64_t start_offset)
      : fd_(fd), offset_(start_offset), bytes_written_(0), buf_used_(0) {
    buf_.resize(kBufferSize);
  }
  ~PosixWritableFile() override { (void)Close(); }

  Status Append(Slice data) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    const char* src = reinterpret_cast<const char*>(data.data());
    size_t size = data.size();

    // Fast path: fits entirely in buffer
    if (buf_used_ + size <= kBufferSize) {
      std::memcpy(buf_.data() + buf_used_, src, size);
      buf_used_ += size;
      return Status::Ok();
    }

    // Flush existing buffer first
    if (buf_used_ > 0) {
      Status s = FlushBuffer();
      if (!s.ok()) return s;
    }

    // Large writes bypass buffer
    if (size >= kBufferSize / 2) {
      return WriteDirect(src, size);
    }

    std::memcpy(buf_.data(), src, size);
    buf_used_ = size;
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
    if (::fdatasync(fd_) != 0) return ErrnoStatus("fdatasync");
    return Status::Ok();
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    Status s = FlushBuffer();
    int r;
    do {
      r = ::close(fd_);
    } while (r != 0 && errno == EINTR);
    fd_ = -1;
    if (!s.ok()) return s;
    return r == 0 ? Status::Ok() : ErrnoStatus("close");
  }

 private:
  Status WriteDirect(const char* p, size_t rem) {
    while (rem > 0) {
      ssize_t w = ::pwrite(fd_, p, rem, static_cast<off_t>(offset_));
      if (w < 0) {
        if (errno == EINTR) continue;
        return ErrnoStatus("pwrite");
      }
      p += static_cast<size_t>(w);
      rem -= static_cast<size_t>(w);
      offset_ += static_cast<uint64_t>(w);
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

  int fd_ = -1;
  uint64_t offset_ = 0;
  uint64_t bytes_written_ = 0;
  size_t buf_used_ = 0;
  std::vector<char> buf_;
};

// -----------------------------------------------------------------------------
// PosixFileMapping
// -----------------------------------------------------------------------------
class PosixFileMapping : public FileMapping {
 public:
  PosixFileMapping(void* addr, size_t size) : data_(static_cast<const uint8_t*>(addr)), size_(size) {}
  ~PosixFileMapping() override {
    if (data_ && size_ > 0) {
      ::munmap(const_cast<uint8_t*>(data_), size_);
    }
  }
  const void* Data() const override { return data_; }
  size_t Size() const override { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace detail

// -----------------------------------------------------------------------------
// PosixEnv
// -----------------------------------------------------------------------------
Status PosixEnv::NewSequentialFile(const std::string& path,
                                   std::unique_ptr<SequentialFile>* result) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open");
  *result = std::make_unique<detail::PosixSequentialFile>(fd);
  return Status::Ok();
}

Status PosixEnv::NewRandomAccessFile(const std::string& path,
                                     std::unique_ptr<RandomAccessFile>* result) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open");
  *result = std::make_unique<detail::PosixRandomAccessFile>(fd);
  return Status::Ok();
}

Status PosixEnv::NewWritableFile(const std::string& path,
                                 std::unique_ptr<WritableFile>* result) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return detail::ErrnoStatus("open");
  *result = std::make_unique<detail::PosixWritableFile>(fd, 0);
  return Status::Ok();
}

Status PosixEnv::NewAppendableFile(const std::string& path,
                                  std::unique_ptr<WritableFile>* result) {
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return detail::ErrnoStatus("open");
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return detail::ErrnoStatus("fstat");
  }
  uint64_t start = static_cast<uint64_t>(st.st_size);
  *result = std::make_unique<detail::PosixWritableFile>(fd, start);
  return Status::Ok();
}

Status PosixEnv::NewFileMapping(const std::string& path,
                                std::unique_ptr<FileMapping>* result) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open");
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return detail::ErrnoStatus("fstat");
  }
  size_t size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    ::close(fd);
    *result = std::make_unique<detail::PosixFileMapping>(nullptr, 0);
    return Status::Ok();
  }
  void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (addr == MAP_FAILED) return detail::ErrnoStatus("mmap");
  *result = std::make_unique<detail::PosixFileMapping>(addr, size);
  return Status::Ok();
}

Status PosixEnv::FileExists(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    if (errno == ENOENT) return Status::NotFound("file does not exist");
    return detail::ErrnoStatus("stat");
  }
  return Status::Ok();
}

Status PosixEnv::GetFileSize(const std::string& path, uint64_t* size) {
  if (!size) return Status::InvalidArgument("size is null");
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return detail::ErrnoStatus("stat");
  *size = static_cast<uint64_t>(st.st_size);
  return Status::Ok();
}

Status PosixEnv::DeleteFile(const std::string& path) {
  if (::unlink(path.c_str()) != 0) return detail::ErrnoStatus("unlink");
  return Status::Ok();
}

Status PosixEnv::RenameFile(const std::string& src, const std::string& dst) {
  if (::rename(src.c_str(), dst.c_str()) != 0) return detail::ErrnoStatus("rename");
  return Status::Ok();
}

Status PosixEnv::CreateDirIfMissing(const std::string& path) {
  if (path.empty()) return Status::Ok();
  std::string p;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!p.empty()) {
        if (::mkdir(p.c_str(), 0755) != 0 && errno != EEXIST)
          return detail::ErrnoStatus("mkdir");
      }
    }
    if (i < path.size()) p += path[i];
  }
  return Status::Ok();
}

Status PosixEnv::SyncDir(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open dir");
  int r;
  do {
    r = ::fsync(fd);
  } while (r != 0 && errno == EINTR);
  int saved = errno;
  ::close(fd);
  if (r != 0) return Status::IOError(std::string("fsync dir: ") + std::strerror(saved));
  return Status::Ok();
}

#else  // Windows

namespace detail {

class WindowsSequentialFile : public SequentialFile {
 public:
  explicit WindowsSequentialFile(HANDLE h) : h_(h) {}
  ~WindowsSequentialFile() override { (void)Close(); }

  Status Read(size_t n, Slice* result) override {
    if (h_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    buf_.resize(n);
    DWORD read_bytes = 0;
    if (!ReadFile(h_, buf_.data(), static_cast<DWORD>(n), &read_bytes, NULL)) {
      return Status::IOError("ReadFile failed");
    }
    buf_.resize(static_cast<size_t>(read_bytes));
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Skip(uint64_t n) override {
    if (h_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(n);
    if (!SetFilePointerEx(h_, li, NULL, FILE_CURRENT)) {
      return Status::IOError("SetFilePointerEx failed");
    }
    return Status::Ok();
  }

  Status Close() override {
    if (h_ != INVALID_HANDLE_VALUE) {
      CloseHandle(h_);
      h_ = INVALID_HANDLE_VALUE;
    }
    return Status::Ok();
  }

 private:
  HANDLE h_ = INVALID_HANDLE_VALUE;
  std::vector<char> buf_;
};

class WindowsRandomAccessFile : public RandomAccessFile {
 public:
  explicit WindowsRandomAccessFile(HANDLE h) : h_(h) {}
  ~WindowsRandomAccessFile() override { (void)Close(); }

  Status Read(uint64_t offset, size_t n, Slice* result) override {
    if (h_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    buf_.resize(n);
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    ov.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
    DWORD read_bytes = 0;
    if (!ReadFile(h_, buf_.data(), static_cast<DWORD>(n), &read_bytes, &ov)) {
      DWORD err = GetLastError();
      if (err == ERROR_HANDLE_EOF) {
        read_bytes = 0;
      } else {
        return Status::IOError("ReadFile failed");
      }
    }
    buf_.resize(static_cast<size_t>(read_bytes));
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Close() override {
    if (h_ != INVALID_HANDLE_VALUE) {
      CloseHandle(h_);
      h_ = INVALID_HANDLE_VALUE;
    }
    return Status::Ok();
  }

 private:
  HANDLE h_ = INVALID_HANDLE_VALUE;
  mutable std::vector<char> buf_;
};

class WindowsWritableFile : public WritableFile {
 public:
  static constexpr size_t kBufferSize = 65536;

  WindowsWritableFile(HANDLE h, uint64_t start_offset)
      : h_(h), offset_(start_offset), bytes_written_(0), buf_used_(0) {
    buf_.resize(kBufferSize);
  }
  ~WindowsWritableFile() override { (void)Close(); }

  Status Append(Slice data) override {
    if (h_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    const char* src = reinterpret_cast<const char*>(data.data());
    size_t size = data.size();

    // Fast path: fits entirely in buffer
    if (buf_used_ + size <= kBufferSize) {
      std::memcpy(buf_.data() + buf_used_, src, size);
      buf_used_ += size;
      return Status::Ok();
    }

    // Flush existing buffer first
    if (buf_used_ > 0) {
      Status s = FlushBuffer();
      if (!s.ok()) return s;
    }

    // Large writes bypass buffer
    if (size >= kBufferSize / 2) {
      return WriteDirect(src, size);
    }

    std::memcpy(buf_.data(), src, size);
    buf_used_ = size;
    return Status::Ok();
  }

  uint64_t BytesWritten() const override { return bytes_written_ + buf_used_; }

  Status Flush() override {
    if (h_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    return FlushBuffer();
  }

  Status Sync() override {
    if (h_ == INVALID_HANDLE_VALUE) return Status::InvalidArgument("file closed");
    Status s = FlushBuffer();
    if (!s.ok()) return s;
    return FlushFileBuffers(h_) ? Status::Ok() : Status::IOError("FlushFileBuffers failed");
  }

  Status Close() override {
    if (h_ != INVALID_HANDLE_VALUE) {
      Status s = FlushBuffer();
      CloseHandle(h_);
      h_ = INVALID_HANDLE_VALUE;
      return s;
    }
    return Status::Ok();
  }

 private:
  Status WriteDirect(const char* data, size_t size) {
    while (size > 0) {
      DWORD to_write = static_cast<DWORD>(std::min<size_t>(size, 64 * 1024 * 1024));
      DWORD written = 0;
      OVERLAPPED ov{};
      ov.Offset = static_cast<DWORD>(offset_ & 0xFFFFFFFF);
      ov.OffsetHigh = static_cast<DWORD>((offset_ >> 32) & 0xFFFFFFFF);
      if (!WriteFile(h_, data, to_write, &written, &ov) || written != to_write) {
        return Status::IOError("WriteFile failed");
      }
      offset_ += written;
      bytes_written_ += written;
      data += written;
      size -= written;
    }
    return Status::Ok();
  }

  Status FlushBuffer() {
    if (buf_used_ == 0) return Status::Ok();
    Status s = WriteDirect(buf_.data(), buf_used_);
    buf_used_ = 0;
    return s;
  }

  HANDLE h_ = INVALID_HANDLE_VALUE;
  uint64_t offset_ = 0;
  uint64_t bytes_written_ = 0;
  size_t buf_used_ = 0;
  std::vector<char> buf_;
};

class WindowsFileMapping : public FileMapping {
 public:
  WindowsFileMapping(HANDLE h_file, HANDLE h_map, const void* data, size_t size)
      : h_file_(h_file), h_map_(h_map), data_(data), size_(size) {}
  ~WindowsFileMapping() override {
    if (data_) UnmapViewOfFile(data_);
    if (h_map_) CloseHandle(h_map_);
    if (h_file_ != INVALID_HANDLE_VALUE) CloseHandle(h_file_);
  }
  const void* Data() const override { return data_; }
  size_t Size() const override { return size_; }

 private:
  HANDLE h_file_ = INVALID_HANDLE_VALUE;
  HANDLE h_map_ = NULL;
  const void* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace detail

Status PosixEnv::NewSequentialFile(const std::string& path,
                                   std::unique_ptr<SequentialFile>* result) {
  std::filesystem::path fs_path(path);
  HANDLE h = CreateFileW(fs_path.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path);
  *result = std::make_unique<detail::WindowsSequentialFile>(h);
  return Status::Ok();
}

Status PosixEnv::NewRandomAccessFile(const std::string& path,
                                     std::unique_ptr<RandomAccessFile>* result) {
  std::filesystem::path fs_path(path);
  HANDLE h = CreateFileW(fs_path.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path);
  *result = std::make_unique<detail::WindowsRandomAccessFile>(h);
  return Status::Ok();
}

Status PosixEnv::NewWritableFile(const std::string& path,
                                 std::unique_ptr<WritableFile>* result) {
  std::filesystem::path fs_path(path);
  HANDLE h = CreateFileW(fs_path.c_str(), GENERIC_WRITE,
                         FILE_SHARE_READ, NULL, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path);
  *result = std::make_unique<detail::WindowsWritableFile>(h, 0);
  return Status::Ok();
}

Status PosixEnv::NewAppendableFile(const std::string& path,
                                   std::unique_ptr<WritableFile>* result) {
  std::filesystem::path fs_path(path);
  HANDLE h = CreateFileW(fs_path.c_str(), GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ, NULL, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path);
  LARGE_INTEGER li;
  if (!GetFileSizeEx(h, &li)) {
    CloseHandle(h);
    return Status::IOError("GetFileSizeEx failed: " + path);
  }
  *result = std::make_unique<detail::WindowsWritableFile>(h, static_cast<uint64_t>(li.QuadPart));
  return Status::Ok();
}

Status PosixEnv::NewFileMapping(const std::string& path,
                                std::unique_ptr<FileMapping>* result) {
  std::filesystem::path fs_path(path);
  HANDLE h_file = CreateFileW(fs_path.c_str(), GENERIC_READ,
                              FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
  if (h_file == INVALID_HANDLE_VALUE) return Status::IOError("open failed: " + path);
  LARGE_INTEGER li;
  if (!GetFileSizeEx(h_file, &li)) {
    CloseHandle(h_file);
    return Status::IOError("GetFileSizeEx failed: " + path);
  }
  size_t size = static_cast<size_t>(li.QuadPart);
  if (size == 0) {
    *result = std::make_unique<detail::WindowsFileMapping>(h_file, static_cast<HANDLE>(NULL), nullptr, 0);
    return Status::Ok();
  }
  HANDLE h_map = CreateFileMappingW(h_file, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!h_map) {
    CloseHandle(h_file);
    return Status::IOError("CreateFileMapping failed: " + path);
  }
  void* data = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, size);
  if (!data) {
    CloseHandle(h_map);
    CloseHandle(h_file);
    return Status::IOError("MapViewOfFile failed: " + path);
  }
  *result = std::make_unique<detail::WindowsFileMapping>(h_file, h_map, data, size);
  return Status::Ok();
}

Status PosixEnv::FileExists(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::exists(std::filesystem::path(path), ec)) {
    return Status::Ok();
  }
  return Status::NotFound("file does not exist");
}

Status PosixEnv::GetFileSize(const std::string& path, uint64_t* size) {
  if (!size) return Status::InvalidArgument("size is null");
  std::error_code ec;
  auto s = std::filesystem::file_size(std::filesystem::path(path), ec);
  if (ec) return Status::IOError("file_size failed: " + ec.message());
  *size = static_cast<uint64_t>(s);
  return Status::Ok();
}

Status PosixEnv::DeleteFile(const std::string& path) {
  std::error_code ec;
  if (!std::filesystem::remove(std::filesystem::path(path), ec)) {
    if (ec) return Status::IOError("remove failed: " + ec.message());
    return Status::NotFound("file does not exist");
  }
  return Status::Ok();
}

Status PosixEnv::RenameFile(const std::string& src, const std::string& dst) {
  std::error_code ec;
  std::filesystem::rename(std::filesystem::path(src), std::filesystem::path(dst), ec);
  if (ec) return Status::IOError("rename failed: " + ec.message());
  return Status::Ok();
}

Status PosixEnv::CreateDirIfMissing(const std::string& path) {
  if (path.empty()) return Status::Ok();
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path), ec);
  if (ec) return Status::IOError("create_directories failed: " + ec.message());
  return Status::Ok();
}

Status PosixEnv::SyncDir(const std::string&) {
  return Status::Ok();
}

#endif

}  // namespace pomai
