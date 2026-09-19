// src/storage/palloc_io.cc — Zero-libc, palloc-backed I/O implementation
//
// Platform-specific implementation using low-level file I/O.
// POSIX on Linux/macOS, Windows API on Windows.
// All buffers allocated from palloc with 4096-byte alignment.
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "palloc_io.h"

#include "utils/palloc_compat.h"
#include "utils/logging.h"

#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

namespace pomai::storage {

#ifdef _WIN32
// =============================================================================
// Windows Implementation
// =============================================================================

namespace {
inline void NormalizeWinPath(const char* in, char* out, size_t max_len) {
    if (!in || !out || max_len == 0) return;
    size_t r = 0;
    size_t w = 0;
    while (in[r] != '\0' && w + 1 < max_len) {
        char c = (in[r] == '/') ? '\\' : in[r];
        if (c == '\\' && w > 0 && out[w - 1] == '\\') {
            if (!(w == 1 && out[0] == '\\')) {
                r++;
                continue;
            }
        }
        out[w++] = c;
        r++;
    }
    out[w] = '\0';
}
} // namespace

class WindowsPallocSequentialFile : public PallocSequentialFile {
public:
    explicit WindowsPallocSequentialFile(HANDLE handle) : handle_(handle) {
        buffer_capacity_ = kDefaultIoBufferSize;
        read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
    }

    ~WindowsPallocSequentialFile() override {
        (void)Close();
        if (read_buffer_) {
            palloc_free(read_buffer_);
        }
    }

    Status Read(size_t n, Slice* result) override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        if (n > buffer_capacity_) {
            if (read_buffer_) palloc_free(read_buffer_);
            buffer_capacity_ = n;
            read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
        }

        DWORD bytes_read = 0;
        BOOL success = ::ReadFile(handle_, read_buffer_, static_cast<DWORD>(n), &bytes_read, nullptr);
        if (!success) {
            return Status::IOError("Read failed");
        }

        buffer_valid_ = bytes_read;
        buffer_pos_ = 0;
        *result = Slice(static_cast<const char*>(read_buffer_), buffer_valid_);
        return Status::Ok();
    }

    Status Skip(uint64_t n) override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        LARGE_INTEGER distance;
        distance.QuadPart = static_cast<LONGLONG>(n);
        LARGE_INTEGER new_pos;
        if (!::SetFilePointerEx(handle_, distance, &new_pos, FILE_CURRENT)) {
            return Status::IOError("Seek failed");
        }
        buffer_pos_ = 0;
        buffer_valid_ = 0;
        return Status::Ok();
    }

    Status Close() override {
        if (handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
        return Status::Ok();
    }

private:
    HANDLE handle_{INVALID_HANDLE_VALUE};
};

Status PallocSequentialFile::Open(const char* path, alloc::UniquePtr<PallocSequentialFile>* out) {
    char norm_path[MAX_PATH];
    NormalizeWinPath(path, norm_path, sizeof(norm_path));
    POMAI_LOG_INFO("PallocSequentialFile::Open: path='{}'", norm_path);
    
    HANDLE handle = ::CreateFileA(norm_path, GENERIC_READ,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        DWORD err = ::GetLastError();
        char err_msg[256];
        DWORD len = ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 
                                      nullptr, err, 0, err_msg, sizeof(err_msg) - 1, nullptr);
        if (len > 0) {
            while (len > 0 && (err_msg[len-1] == '\r' || err_msg[len-1] == '\n')) {
                err_msg[--len] = '\0';
            }
            POMAI_LOG_ERROR("PallocSequentialFile::Open FAILED for '{}': Windows error={}, message='{}'", norm_path, err, err_msg);
        } else {
            POMAI_LOG_ERROR("PallocSequentialFile::Open FAILED for '{}': Windows error={}", norm_path, err);
        }
        return Status::IOError("Failed to open file");
    }

    POMAI_LOG_INFO("PallocSequentialFile::Open SUCCESS for '{}'", norm_path);
    auto file = alloc::UniquePtr<WindowsPallocSequentialFile>::Make(nullptr, handle);
    if (!file) {
        ::CloseHandle(handle);
        return Status::IOError("Failed to allocate WindowsPallocSequentialFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

class WindowsPallocRandomAccessFile : public PallocRandomAccessFile {
public:
    explicit WindowsPallocRandomAccessFile(HANDLE handle) : handle_(handle) {
        buffer_capacity_ = kDefaultIoBufferSize;
        read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
    }

    ~WindowsPallocRandomAccessFile() override {
        (void)Close();
        if (read_buffer_) {
            palloc_free(read_buffer_);
        }
    }

    Status Read(uint64_t offset, size_t n, Slice* result) override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        if (n > buffer_capacity_) {
            if (read_buffer_) palloc_free(read_buffer_);
            buffer_capacity_ = n;
            read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
        }

        LARGE_INTEGER pos;
        pos.QuadPart = static_cast<LONGLONG>(offset);
        if (!::SetFilePointerEx(handle_, pos, nullptr, FILE_BEGIN)) {
            return Status::IOError("Seek failed");
        }

        DWORD bytes_read = 0;
        BOOL success = ::ReadFile(handle_, read_buffer_, static_cast<DWORD>(n), &bytes_read, nullptr);
        if (!success) {
            return Status::IOError("Read failed");
        }

        *result = Slice(static_cast<const char*>(read_buffer_), bytes_read);
        return Status::Ok();
    }

    Status Close() override {
        if (handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
        return Status::Ok();
    }

private:
    HANDLE handle_{INVALID_HANDLE_VALUE};
};

Status PallocRandomAccessFile::Open(const char* path, alloc::UniquePtr<PallocRandomAccessFile>* out) {
    char norm_path[MAX_PATH];
    NormalizeWinPath(path, norm_path, sizeof(norm_path));
    POMAI_LOG_INFO("PallocRandomAccessFile::Open: path='{}'", norm_path);
    
    HANDLE handle = ::CreateFileA(norm_path, GENERIC_READ,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        DWORD err = ::GetLastError();
        char err_msg[256];
        DWORD len = ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 
                                      nullptr, err, 0, err_msg, sizeof(err_msg) - 1, nullptr);
        if (len > 0) {
            while (len > 0 && (err_msg[len-1] == '\r' || err_msg[len-1] == '\n')) {
                err_msg[--len] = '\0';
            }
            POMAI_LOG_ERROR("PallocRandomAccessFile::Open FAILED for '{}': Windows error={}, message='{}'", norm_path, err, err_msg);
        } else {
            POMAI_LOG_ERROR("PallocRandomAccessFile::Open FAILED for '{}': Windows error={}", norm_path, err);
        }
        return Status::IOError("Failed to open file");
    }

    POMAI_LOG_INFO("PallocRandomAccessFile::Open SUCCESS for '{}'", norm_path);
    auto file = alloc::UniquePtr<WindowsPallocRandomAccessFile>::Make(nullptr, handle);
    if (!file) {
        ::CloseHandle(handle);
        return Status::IOError("Failed to allocate WindowsPallocRandomAccessFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

class WindowsPallocWritableFile : public PallocWritableFile {
public:
    explicit WindowsPallocWritableFile(HANDLE handle) : handle_(handle) {
        buffer_capacity_ = kDefaultIoBufferSize;
        write_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
    }

    ~WindowsPallocWritableFile() override {
        (void)Close();
        if (write_buffer_) {
            palloc_free(write_buffer_);
        }
    }

    Status Append(Slice data) override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        if (buffer_pos_ + data.size() > buffer_capacity_) {
            auto st = Flush();
            if (!st.ok()) return st;
        }

        if (data.size() > buffer_capacity_) {
            DWORD bytes_written = 0;
            BOOL success = ::WriteFile(handle_, data.data(), static_cast<DWORD>(data.size()),
                                        &bytes_written, nullptr);
            if (!success || bytes_written != data.size()) {
                return Status::IOError("Write failed");
            }
            bytes_written_ += data.size();
        } else {
            ::memcpy(static_cast<char*>(write_buffer_) + buffer_pos_, data.data(), data.size());
            buffer_pos_ += data.size();
        }

        return Status::Ok();
    }

    Status Pwrite(uint64_t offset, Slice data) override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        auto st = Flush();
        if (!st.ok()) return st;

        LARGE_INTEGER saved_pos;
        LARGE_INTEGER zero;
        zero.QuadPart = 0;
        if (!::SetFilePointerEx(handle_, zero, &saved_pos, FILE_CURRENT)) {
            return Status::IOError("Failed to query file position");
        }

        LARGE_INTEGER target_pos;
        target_pos.QuadPart = static_cast<LONGLONG>(offset);
        if (!::SetFilePointerEx(handle_, target_pos, nullptr, FILE_BEGIN)) {
            return Status::IOError("Failed to seek for pwrite");
        }

        DWORD bytes_written = 0;
        BOOL success = ::WriteFile(handle_, data.data(), static_cast<DWORD>(data.size()),
                                    &bytes_written, nullptr);
        ::SetFilePointerEx(handle_, saved_pos, nullptr, FILE_BEGIN);

        if (!success || bytes_written != data.size()) {
            return Status::IOError("Pwrite failed");
        }
        return Status::Ok();
    }

    uint64_t BytesWritten() const override {
        return bytes_written_ + buffer_pos_;
    }

    Status Flush() override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        if (buffer_pos_ > 0) {
            DWORD bytes_written = 0;
            BOOL success = ::WriteFile(handle_, write_buffer_, static_cast<DWORD>(buffer_pos_),
                                        &bytes_written, nullptr);
            if (!success || bytes_written != buffer_pos_) {
                return Status::IOError("Flush write failed");
            }
            bytes_written_ += buffer_pos_;
            buffer_pos_ = 0;
        }
        return Status::Ok();
    }

    Status Sync() override {
        if (handle_ == INVALID_HANDLE_VALUE) return Status::IOError("File closed");

        auto st = Flush();
        if (!st.ok()) return st;

        if (!::FlushFileBuffers(handle_)) {
            return Status::IOError("FlushFileBuffers failed");
        }
        return Status::Ok();
    }

    Status Close() override {
        if (handle_ != INVALID_HANDLE_VALUE) {
            auto st = Sync();
            ::CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
            return st;
        }
        return Status::Ok();
    }

private:
    HANDLE handle_{INVALID_HANDLE_VALUE};
};

Status PallocWritableFile::Create(const char* path, alloc::UniquePtr<PallocWritableFile>* out) {
    char norm_path[MAX_PATH];
    NormalizeWinPath(path, norm_path, sizeof(norm_path));
    POMAI_LOG_INFO("PallocWritableFile::Create: path='{}'", norm_path);
    
    HANDLE handle = ::CreateFileA(norm_path, GENERIC_WRITE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        DWORD err = ::GetLastError();
        char err_msg[256];
        DWORD len = ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 
                                      nullptr, err, 0, err_msg, sizeof(err_msg) - 1, nullptr);
        if (len > 0) {
            while (len > 0 && (err_msg[len-1] == '\r' || err_msg[len-1] == '\n')) {
                err_msg[--len] = '\0';
            }
            POMAI_LOG_ERROR("PallocWritableFile::Create FAILED for '{}': Windows error={}, message='{}'", norm_path, err, err_msg);
        } else {
            POMAI_LOG_ERROR("PallocWritableFile::Create FAILED for '{}': Windows error={}", norm_path, err);
        }
        return Status::IOError("Failed to create file");
    }

    POMAI_LOG_INFO("PallocWritableFile::Create SUCCESS for '{}'", norm_path);
    auto file = alloc::UniquePtr<WindowsPallocWritableFile>::Make(nullptr, handle);
    if (!file) {
        ::CloseHandle(handle);
        return Status::IOError("Failed to allocate WindowsPallocWritableFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

Status PallocWritableFile::OpenAppend(const char* path, alloc::UniquePtr<PallocWritableFile>* out) {
    char norm_path[MAX_PATH];
    NormalizeWinPath(path, norm_path, sizeof(norm_path));
    POMAI_LOG_INFO("PallocWritableFile::OpenAppend: path='{}'", norm_path);
    
    HANDLE handle = ::CreateFileA(norm_path, GENERIC_WRITE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        DWORD err = ::GetLastError();
        char err_msg[256];
        DWORD len = ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 
                                      nullptr, err, 0, err_msg, sizeof(err_msg) - 1, nullptr);
        if (len > 0) {
            while (len > 0 && (err_msg[len-1] == '\r' || err_msg[len-1] == '\n')) {
                err_msg[--len] = '\0';
            }
            POMAI_LOG_ERROR("PallocWritableFile::OpenAppend FAILED for '{}': Windows error={}, message='{}'", norm_path, err, err_msg);
        } else {
            POMAI_LOG_ERROR("PallocWritableFile::OpenAppend FAILED for '{}': Windows error={}", norm_path, err);
        }
        return Status::IOError("Failed to open file for append");
    }

    POMAI_LOG_INFO("PallocWritableFile::OpenAppend SUCCESS for '{}'", norm_path);

    // Seek to end
    LARGE_INTEGER distance;
    distance.QuadPart = 0;
    if (!::SetFilePointerEx(handle, distance, nullptr, FILE_END)) {
        DWORD err = ::GetLastError();
        POMAI_LOG_ERROR("PallocWritableFile::OpenAppend SetFilePointerEx FAILED for '{}': Windows error={}", norm_path, err);
        ::CloseHandle(handle);
        return Status::IOError("Seek to end failed");
    }

    auto file = alloc::UniquePtr<WindowsPallocWritableFile>::Make(nullptr, handle);
    if (!file) {
        ::CloseHandle(handle);
        return Status::IOError("Failed to allocate WindowsPallocWritableFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

class WindowsPallocFileMapping : public PallocFileMapping {
public:
    static Status Map(const char* path, alloc::UniquePtr<PallocFileMapping>* out) {
        char norm_path[MAX_PATH];
        NormalizeWinPath(path, norm_path, sizeof(norm_path));

        HANDLE file_handle = ::CreateFileA(norm_path, GENERIC_READ,
                                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            return Status::IOError("Failed to open file for mapping");
        }

        LARGE_INTEGER file_size;
        if (!::GetFileSizeEx(file_handle, &file_size)) {
            ::CloseHandle(file_handle);
            return Status::IOError("Failed to get file size");
        }

        size_t size = static_cast<size_t>(file_size.QuadPart);
        HANDLE mapping_handle = ::CreateFileMappingA(file_handle, nullptr, PAGE_READONLY,
                                                     0, 0, nullptr);
        ::CloseHandle(file_handle);

        if (mapping_handle == nullptr) {
            return Status::IOError("CreateFileMapping failed");
        }

        void* mapped = ::MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, size);
        ::CloseHandle(mapping_handle);

        if (mapped == nullptr) {
            return Status::IOError("MapViewOfFile failed");
        }

        auto mapping = alloc::UniquePtr<WindowsPallocFileMapping>::Make(nullptr, mapped, size);
        if (!mapping) {
            ::UnmapViewOfFile(mapped);
            return Status::IOError("Failed to allocate WindowsPallocFileMapping");
        }
        *out = std::move(mapping);
        return Status::Ok();
    }

    WindowsPallocFileMapping(void* data, size_t size)
        : mapped_data_(data), mapped_size_(size) {}

    ~WindowsPallocFileMapping() override {
        if (mapped_data_ && mapped_size_ > 0) {
            ::UnmapViewOfFile(mapped_data_);
        }
    }

    const void* Data() const override { return mapped_data_; }
    size_t Size() const override { return mapped_size_; }

private:
    void* mapped_data_{nullptr};
    size_t mapped_size_{0};
};

Status PallocFileMapping::Map(const char* path, alloc::UniquePtr<PallocFileMapping>* out) {
    return WindowsPallocFileMapping::Map(path, out);
}

#else
// =============================================================================
// POSIX Implementation
// =============================================================================

class PosixPallocSequentialFile : public PallocSequentialFile {
public:
    explicit PosixPallocSequentialFile(int fd) : fd_(fd) {
        buffer_capacity_ = kDefaultIoBufferSize;
        read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
    }

    ~PosixPallocSequentialFile() override {
        (void)Close();
        if (read_buffer_) {
            palloc_free(read_buffer_);
        }
    }

    Status Read(size_t n, Slice* result) override {
        if (fd_ < 0) return Status::IOError("File closed");

        if (n > buffer_capacity_) {
            if (read_buffer_) palloc_free(read_buffer_);
            buffer_capacity_ = n;
            read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
        }

        ssize_t bytes_read = ::read(fd_, read_buffer_, n);
        if (bytes_read < 0) {
            return Status::IOError("Read failed");
        }

        buffer_valid_ = static_cast<size_t>(bytes_read);
        buffer_pos_ = 0;
        *result = Slice(static_cast<const char*>(read_buffer_), buffer_valid_);
        return Status::Ok();
    }

    Status Skip(uint64_t n) override {
        if (fd_ < 0) return Status::IOError("File closed");

        off_t offset = ::lseek(fd_, static_cast<off_t>(n), SEEK_CUR);
        if (offset < 0) {
            return Status::IOError("Seek failed");
        }
        buffer_pos_ = 0;
        buffer_valid_ = 0;
        return Status::Ok();
    }

    Status Close() override {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        return Status::Ok();
    }

private:
    int fd_{-1};
};

Status PallocSequentialFile::Open(const char* path, alloc::UniquePtr<PallocSequentialFile>* out) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return Status::IOError("Failed to open file");
    }

    auto file = alloc::UniquePtr<PosixPallocSequentialFile>::Make(nullptr, fd);
    if (!file) {
        ::close(fd);
        return Status::IOError("Failed to allocate PosixPallocSequentialFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

class PosixPallocRandomAccessFile : public PallocRandomAccessFile {
public:
    explicit PosixPallocRandomAccessFile(int fd) : fd_(fd) {
        buffer_capacity_ = kDefaultIoBufferSize;
        read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
    }

    ~PosixPallocRandomAccessFile() override {
        (void)Close();
        if (read_buffer_) {
            palloc_free(read_buffer_);
        }
    }

    Status Read(uint64_t offset, size_t n, Slice* result) override {
        if (fd_ < 0) return Status::IOError("File closed");

        if (n > buffer_capacity_) {
            if (read_buffer_) palloc_free(read_buffer_);
            buffer_capacity_ = n;
            read_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
        }

        ssize_t bytes_read = ::pread(fd_, read_buffer_, n, static_cast<off_t>(offset));
        if (bytes_read < 0) {
            return Status::IOError("Pread failed");
        }

        *result = Slice(static_cast<const char*>(read_buffer_), static_cast<size_t>(bytes_read));
        return Status::Ok();
    }

    Status Close() override {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        return Status::Ok();
    }

private:
    int fd_{-1};
};

Status PallocRandomAccessFile::Open(const char* path, alloc::UniquePtr<PallocRandomAccessFile>* out) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return Status::IOError("Failed to open file");
    }

    auto file = alloc::UniquePtr<PosixPallocRandomAccessFile>::Make(nullptr, fd);
    if (!file) {
        ::close(fd);
        return Status::IOError("Failed to allocate PosixPallocRandomAccessFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

class PosixPallocWritableFile : public PallocWritableFile {
public:
    explicit PosixPallocWritableFile(int fd) : fd_(fd) {
        buffer_capacity_ = kDefaultIoBufferSize;
        write_buffer_ = palloc_malloc_aligned(buffer_capacity_, kIoAlignment);
    }

    ~PosixPallocWritableFile() override {
        (void)Close();
        if (write_buffer_) {
            palloc_free(write_buffer_);
        }
    }

    Status Append(Slice data) override {
        if (fd_ < 0) return Status::IOError("File closed");

        if (buffer_pos_ + data.size() > buffer_capacity_) {
            auto st = Flush();
            if (!st.ok()) return st;
        }

        if (data.size() > buffer_capacity_) {
            ssize_t written = ::write(fd_, data.data(), data.size());
            if (written < 0 || static_cast<size_t>(written) != data.size()) {
                return Status::IOError("Write failed");
            }
            bytes_written_ += data.size();
        } else {
            ::memcpy(static_cast<char*>(write_buffer_) + buffer_pos_, data.data(), data.size());
            buffer_pos_ += data.size();
        }

        return Status::Ok();
    }

    Status Pwrite(uint64_t offset, Slice data) override {
        if (fd_ < 0) return Status::IOError("File closed");
        auto st = Flush();
        if (!st.ok()) return st;

        ssize_t written = ::pwrite(fd_, data.data(), data.size(), static_cast<off_t>(offset));
        if (written < 0 || static_cast<size_t>(written) != data.size()) {
            return Status::IOError("Pwrite failed");
        }
        return Status::Ok();
    }

    uint64_t BytesWritten() const override {
        return bytes_written_ + buffer_pos_;
    }

    Status Flush() override {
        if (fd_ < 0) return Status::IOError("File closed");

        if (buffer_pos_ > 0) {
            ssize_t written = ::write(fd_, write_buffer_, buffer_pos_);
            if (written < 0 || static_cast<size_t>(written) != buffer_pos_) {
                return Status::IOError("Flush write failed");
            }
            bytes_written_ += buffer_pos_;
            buffer_pos_ = 0;
        }
        return Status::Ok();
    }

    Status Sync() override {
        if (fd_ < 0) return Status::IOError("File closed");

        auto st = Flush();
        if (!st.ok()) return st;

        if (::fsync(fd_) < 0) {
            return Status::IOError("Fsync failed");
        }
        return Status::Ok();
    }

    Status Close() override {
        if (fd_ >= 0) {
            auto st = Sync();
            ::close(fd_);
            fd_ = -1;
            return st;
        }
        return Status::Ok();
    }

private:
    int fd_{-1};
};

Status PallocWritableFile::Create(const char* path, alloc::UniquePtr<PallocWritableFile>* out) {
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return Status::IOError("Failed to create file");
    }

    auto file = alloc::UniquePtr<PosixPallocWritableFile>::Make(nullptr, fd);
    if (!file) {
        ::close(fd);
        return Status::IOError("Failed to allocate PosixPallocWritableFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

Status PallocWritableFile::OpenAppend(const char* path, alloc::UniquePtr<PallocWritableFile>* out) {
    int fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        return Status::IOError("Failed to open file for append");
    }

    auto file = alloc::UniquePtr<PosixPallocWritableFile>::Make(nullptr, fd);
    if (!file) {
        ::close(fd);
        return Status::IOError("Failed to allocate PosixPallocWritableFile");
    }
    *out = std::move(file);
    return Status::Ok();
}

class PosixPallocFileMapping : public PallocFileMapping {
public:
    static Status Map(const char* path, alloc::UniquePtr<PallocFileMapping>* out) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) {
            return Status::IOError("Failed to open file for mapping");
        }

        struct stat st;
        if (::fstat(fd, &st) < 0) {
            ::close(fd);
            return Status::IOError("Failed to stat file");
        }

        size_t size = static_cast<size_t>(st.st_size);
        void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);

        if (mapped == MAP_FAILED) {
            return Status::IOError("mmap failed");
        }

        auto mapping = alloc::UniquePtr<PosixPallocFileMapping>::Make(nullptr, mapped, size);
        if (!mapping) {
            ::munmap(mapped, size);
            return Status::IOError("Failed to allocate PosixPallocFileMapping");
        }
        *out = std::move(mapping);
        return Status::Ok();
    }

    PosixPallocFileMapping(void* data, size_t size)
        : mapped_data_(data), mapped_size_(size) {}

    ~PosixPallocFileMapping() override {
        if (mapped_data_ && mapped_size_ > 0) {
            ::munmap(mapped_data_, mapped_size_);
        }
    }

    const void* Data() const override { return mapped_data_; }
    size_t Size() const override { return mapped_size_; }

private:
    void* mapped_data_{nullptr};
    size_t mapped_size_{0};
};

Status PallocFileMapping::Map(const char* path, alloc::UniquePtr<PallocFileMapping>* out) {
    return PosixPallocFileMapping::Map(path, out);
}

#endif

// Base class destructors
PallocSequentialFile::~PallocSequentialFile() = default;
PallocRandomAccessFile::~PallocRandomAccessFile() = default;
PallocWritableFile::~PallocWritableFile() = default;
PallocFileMapping::~PallocFileMapping() = default;

// =============================================================================
// Filesystem operations
// =============================================================================

#ifdef _WIN32

Status PallocFilesystem::FileExists(const char* path) {
    char normalized_path[MAX_PATH];
    NormalizeWinPath(path, normalized_path, sizeof(normalized_path));
    
    DWORD attrs = ::GetFileAttributesA(normalized_path);
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        DWORD err = ::GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
            return Status::NotFound("File not found");
        }
        return Status::IOError("GetFileAttributes failed");
    }
    return Status::Ok();
}

Status PallocFilesystem::GetFileSize(const char* path, uint64_t* size) {
    char normalized_path[MAX_PATH];
    NormalizeWinPath(path, normalized_path, sizeof(normalized_path));

    WIN32_FILE_ATTRIBUTE_DATA attrs;
    if (!::GetFileAttributesExA(normalized_path, GetFileExInfoStandard, &attrs)) {
        return Status::IOError("GetFileAttributesEx failed");
    }
    LARGE_INTEGER file_size;
    file_size.HighPart = attrs.nFileSizeHigh;
    file_size.LowPart = attrs.nFileSizeLow;
    *size = static_cast<uint64_t>(file_size.QuadPart);
    return Status::Ok();
}

Status PallocFilesystem::RemoveFile(const char* path) {
    char normalized_path[MAX_PATH];
    NormalizeWinPath(path, normalized_path, sizeof(normalized_path));

    if (!::DeleteFileA(normalized_path)) {
        DWORD err = ::GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
            return Status::Ok(); // Delete succeeded, file already gone
        }
        return Status::IOError("DeleteFile failed");
    }
    return Status::Ok();
}

Status PallocFilesystem::RemoveDirRecursive(const char* path) {
    if (!path || path[0] == '\0') return Status::InvalidArgument("empty path");
    char normalized_path[MAX_PATH];
    NormalizeWinPath(path, normalized_path, sizeof(normalized_path));
    std::error_code ec;
    std::filesystem::remove_all(normalized_path, ec);
    if (ec) {
        return Status::IOError(std::string("remove_all failed: ") + ec.message());
    }
    return Status::Ok();
}

Status PallocFilesystem::CreateDir(const char* path) {
    char normalized_path[MAX_PATH];
    NormalizeWinPath(path, normalized_path, sizeof(normalized_path));

    POMAI_LOG_INFO("PallocFilesystem::CreateDir: path='{}'", normalized_path);

    // Check if directory already exists
    DWORD attrs = ::GetFileAttributesA(normalized_path);
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        POMAI_LOG_INFO("PallocFilesystem::CreateDir: directory already exists");
        return Status::Ok();
    }

    // Try to create parent directories first
    char* last_slash = ::strrchr(normalized_path, '\\');
    if (last_slash && last_slash != normalized_path) {
        // If parent is drive root (e.g. C:\ or D:\) do not recurse
        if (last_slash - normalized_path == 2 && normalized_path[1] == ':') {
            // Drive root already exists
        } else {
            *last_slash = '\0';
            auto st = CreateDir(normalized_path);
            if (!st.ok()) return st;
            *last_slash = '\\';
        }
    }

    if (!::CreateDirectoryA(normalized_path, nullptr)) {
        DWORD err = ::GetLastError();
        if (err == ERROR_ALREADY_EXISTS) {
            POMAI_LOG_INFO("PallocFilesystem::CreateDir: directory already exists (ERROR_ALREADY_EXISTS)");
            return Status::Ok();
        }
        char err_msg[256];
        DWORD len = ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 
                                      nullptr, err, 0, err_msg, sizeof(err_msg) - 1, nullptr);
        if (len > 0) {
            // Remove trailing newlines
            while (len > 0 && (err_msg[len-1] == '\r' || err_msg[len-1] == '\n')) {
                err_msg[--len] = '\0';
            }
            POMAI_LOG_ERROR("PallocFilesystem::CreateDir FAILED for '{}': Windows error={}, message='{}'", normalized_path, err, err_msg);
            return Status::IOError(std::string("CreateDirectory failed for '") + normalized_path + "': " + err_msg);
        }
        POMAI_LOG_ERROR("PallocFilesystem::CreateDir FAILED for '{}': Windows error={}", normalized_path, err);
        return Status::IOError(std::string("CreateDirectory failed for '") + normalized_path + "'");
    }
    POMAI_LOG_INFO("PallocFilesystem::CreateDir: directory created successfully");
    return Status::Ok();
}

Status PallocFilesystem::SyncDir(const char* path) {
    (void)path; // Windows doesn't have a direct equivalent to fsync on directories
    // Directory metadata is synced on handle close automatically
    return Status::Ok();
}

#else

Status PallocFilesystem::FileExists(const char* path) {
    struct stat st;
    if (::stat(path, &st) < 0) {
        if (errno == ENOENT) {
            return Status::NotFound("File not found");
        }
        return Status::IOError("stat failed");
    }
    return Status::Ok();
}

Status PallocFilesystem::GetFileSize(const char* path, uint64_t* size) {
    struct stat st;
    if (::stat(path, &st) < 0) {
        return Status::IOError("stat failed");
    }
    *size = static_cast<uint64_t>(st.st_size);
    return Status::Ok();
}

Status PallocFilesystem::RemoveFile(const char* path) {
    if (::unlink(path) < 0) {
        if (errno == ENOENT) {
            return Status::Ok(); // Delete succeeded, file already gone
        }
        return Status::IOError("unlink failed");
    }
    return Status::Ok();
}

Status PallocFilesystem::RemoveDirRecursive(const char* path) {
    if (!path || path[0] == '\0') return Status::InvalidArgument("empty path");
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        return Status::IOError(std::string("remove_all failed: ") + ec.message());
    }
    return Status::Ok();
}

Status PallocFilesystem::CreateDir(const char* path) {
    // Create directory recursively
    if (::mkdir(path, 0755) < 0) {
        if (errno == EEXIST) {
            return Status::Ok();
        }
        if (errno == ENOENT) {
            // Try to create parent directories
            char parent_path[PATH_MAX];
            ::strncpy(parent_path, path, PATH_MAX - 1);
            parent_path[PATH_MAX - 1] = '\0';
            char* last_slash = ::strrchr(parent_path, '/');
            if (last_slash) {
                *last_slash = '\0';
                auto st = CreateDir(parent_path);
                if (!st.ok()) return st;
                return CreateDir(path);
            }
        }
        return Status::IOError("mkdir failed");
    }
    return Status::Ok();
}

Status PallocFilesystem::SyncDir(const char* path) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return Status::IOError("open dir failed");
    }
    if (::fsync(fd) < 0) {
        ::close(fd);
        return Status::IOError("fsync failed");
    }
    ::close(fd);
    return Status::Ok();
}

#endif

} // namespace pomai::storage
