// include/pio/pio_fs.h — Cross-platform FileSystem operations and factory methods
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include "pio_file.h"
#include "pio_platform.h"
#include "pio_posix.h"
#include "pio_windows.h"

namespace pio {

class FileSystem {
public:
    static Status NewSequentialFile(const std::string& path, std::unique_ptr<SequentialFile>* out) {
        if (!out) return Status::InvalidArgument("null output pointer");
#if defined(PIO_PLATFORM_WINDOWS)
        std::wstring wpath = detail::Utf8ToUtf16(path);
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (h == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
                return Status::NotFound("File not found: " + path);
            }
            return Status::IOError("Failed to open file: " + path);
        }
        *out = std::make_unique<WindowsSequentialFile>(h);
        return Status::Ok();
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT) return Status::NotFound("File not found: " + path);
            return Status::IOError("Failed to open file: " + path);
        }
        *out = std::make_unique<PosixSequentialFile>(fd);
        return Status::Ok();
#endif
    }

    static Status NewRandomAccessFile(const std::string& path, std::unique_ptr<RandomAccessFile>* out) {
        if (!out) return Status::InvalidArgument("null output pointer");
#if defined(PIO_PLATFORM_WINDOWS)
        std::wstring wpath = detail::Utf8ToUtf16(path);
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, NULL);
        if (h == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
                return Status::NotFound("File not found: " + path);
            }
            return Status::IOError("Failed to open file: " + path);
        }
        *out = std::make_unique<WindowsRandomAccessFile>(h);
        return Status::Ok();
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT) return Status::NotFound("File not found: " + path);
            return Status::IOError("Failed to open file: " + path);
        }
        *out = std::make_unique<PosixRandomAccessFile>(fd);
        return Status::Ok();
#endif
    }

    static Status NewWritableFile(const std::string& path, std::unique_ptr<WritableFile>* out) {
        if (!out) return Status::InvalidArgument("null output pointer");
#if defined(PIO_PLATFORM_WINDOWS)
        std::wstring wpath = detail::Utf8ToUtf16(path);
        HANDLE h = CreateFileW(wpath.c_str(), GENERIC_WRITE | GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h == INVALID_HANDLE_VALUE) {
            return Status::IOError("Failed to create file: " + path);
        }
        *out = std::make_unique<WindowsWritableFile>(h);
        return Status::Ok();
#else
        int fd = ::open(path.c_str(), O_TRUNC | O_WRONLY | O_CREAT, 0644);
        if (fd < 0) {
            return Status::IOError("Failed to create file: " + path);
        }
        *out = std::make_unique<PosixWritableFile>(fd);
        return Status::Ok();
#endif
    }

    static Status NewAppendableFile(const std::string& path, std::unique_ptr<WritableFile>* out) {
        if (!out) return Status::InvalidArgument("null output pointer");
#if defined(PIO_PLATFORM_WINDOWS)
        std::wstring wpath = detail::Utf8ToUtf16(path);
        HANDLE h = CreateFileW(wpath.c_str(), FILE_APPEND_DATA | GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h == INVALID_HANDLE_VALUE) {
            return Status::IOError("Failed to open append file: " + path);
        }
        SetFilePointer(h, 0, NULL, FILE_END);
        *out = std::make_unique<WindowsWritableFile>(h);
        return Status::Ok();
#else
        int fd = ::open(path.c_str(), O_APPEND | O_WRONLY | O_CREAT, 0644);
        if (fd < 0) {
            return Status::IOError("Failed to open append file: " + path);
        }
        *out = std::make_unique<PosixWritableFile>(fd);
        return Status::Ok();
#endif
    }

    static Status NewMemoryMappedFile(const std::string& path, std::unique_ptr<MemoryMappedFile>* out) {
        if (!out) return Status::InvalidArgument("null output pointer");
#if defined(PIO_PLATFORM_WINDOWS)
        std::wstring wpath = detail::Utf8ToUtf16(path);
        HANDLE h_file = CreateFileW(wpath.c_str(), GENERIC_READ,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h_file == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
                return Status::NotFound("File not found: " + path);
            }
            return Status::IOError("Failed to open file for mmap: " + path);
        }

        LARGE_INTEGER li;
        if (!GetFileSizeEx(h_file, &li)) {
            CloseHandle(h_file);
            return Status::IOError("Failed to get file size: " + path);
        }
        std::size_t size = static_cast<std::size_t>(li.QuadPart);
        if (size == 0) {
            // Empty files cannot be mapped by CreateFileMapping
            *out = std::make_unique<WindowsMemoryMappedFile>(h_file, static_cast<HANDLE>(nullptr), nullptr, 0);
            return Status::Ok();
        }

        HANDLE h_map = CreateFileMappingW(h_file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!h_map) {
            CloseHandle(h_file);
            return Status::IOError("CreateFileMappingW failed for: " + path);
        }

        void* mapped = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, size);
        if (!mapped) {
            CloseHandle(h_map);
            CloseHandle(h_file);
            return Status::IOError("MapViewOfFile failed for: " + path);
        }

        *out = std::make_unique<WindowsMemoryMappedFile>(h_file, h_map, static_cast<const uint8_t*>(mapped), size);
        return Status::Ok();
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT) return Status::NotFound("File not found: " + path);
            return Status::IOError("Failed to open file for mmap: " + path);
        }

        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            return Status::IOError("fstat failed for: " + path);
        }
        std::size_t size = static_cast<std::size_t>(st.st_size);
        if (size == 0) {
            *out = std::make_unique<PosixMemoryMappedFile>(fd, nullptr, 0);
            return Status::Ok();
        }

        void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            return Status::IOError("mmap failed for: " + path);
        }

        *out = std::make_unique<PosixMemoryMappedFile>(fd, static_cast<const uint8_t*>(mapped), size);
        return Status::Ok();
#endif
    }

    static bool FileExists(const std::string& path) noexcept {
        std::error_code ec;
        return std::filesystem::exists(path, ec);
    }

    static Status GetFileSize(const std::string& path, std::uint64_t* size) noexcept {
        if (!size) return Status::InvalidArgument("null size pointer");
        std::error_code ec;
        auto sz = std::filesystem::file_size(path, ec);
        if (ec) return Status::IOError("GetFileSize failed: " + ec.message());
        *size = static_cast<std::uint64_t>(sz);
        return Status::Ok();
    }

    static Status DeleteFile(const std::string& path) noexcept {
        std::error_code ec;
        if (std::filesystem::remove(path, ec)) return Status::Ok();
        if (!std::filesystem::exists(path, ec)) return Status::Ok();
        return Status::IOError("DeleteFile failed: " + ec.message());
    }

    static Status CreateDir(const std::string& path) noexcept {
        std::error_code ec;
        if (std::filesystem::create_directory(path, ec) || std::filesystem::exists(path, ec)) {
            return Status::Ok();
        }
        return Status::IOError("CreateDir failed: " + ec.message());
    }

    static Status CreateDirAll(const std::string& path) noexcept {
        std::error_code ec;
        if (std::filesystem::create_directories(path, ec) || std::filesystem::exists(path, ec)) {
            return Status::Ok();
        }
        return Status::IOError("CreateDirAll failed: " + ec.message());
    }

    static Status RenameFileAtomic(const std::string& src, const std::string& dst) noexcept {
        std::error_code ec;
        std::filesystem::rename(src, dst, ec);
        if (ec) return Status::IOError("Rename failed: " + ec.message());
        return Status::Ok();
    }

    static Status ListDirectory(const std::string& dir, std::vector<std::string>* result) {
        if (!result) return Status::InvalidArgument("null result pointer");
        result->clear();
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            result->push_back(entry.path().filename().string());
        }
        if (ec) return Status::IOError("ListDirectory failed: " + ec.message());
        return Status::Ok();
    }

    static Status SyncDir(const std::string& dir) noexcept {
#if defined(PIO_PLATFORM_POSIX)
        int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (fd < 0) return Status::IOError("Failed to open directory for sync: " + dir);
        int res = ::fsync(fd);
        ::close(fd);
        return (res == 0) ? Status::Ok() : Status::IOError("Directory fsync failed");
#else
        (void)dir;
        return Status::Ok();
#endif
    }
};

} // namespace pio
