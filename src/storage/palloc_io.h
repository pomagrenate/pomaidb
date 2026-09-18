// src/storage/palloc_io.h — Zero-libc, palloc-backed I/O subsystem
//
// Custom high-performance I/O layer for PomaiDB storage engine.
// All buffers allocated from palloc with 4096-byte alignment for Direct I/O.
// No <memory>, no std::unique_ptr, no libc file streams.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "slice.h"
#include "status.h"
#include "utils/palloc_compat.h"
#include "utils/palloc_smart_ptr.h"

namespace pomai::storage {

// Forward declarations
class PallocSequentialFile;
class PallocRandomAccessFile;
class PallocWritableFile;

// =============================================================================
// PallocSequentialFile — Streaming read-only access with palloc buffers
// =============================================================================

class PallocSequentialFile {
public:
    // Open file for sequential reading
    static Status Open(const char* path, alloc::UniquePtr<PallocSequentialFile>* out);

    virtual ~PallocSequentialFile();

    // Read up to n bytes. Result points to internal palloc buffer.
    // Buffer valid until next Read(), Skip(), or Close().
    virtual Status Read(size_t n, Slice* result) = 0;
    virtual Status Skip(uint64_t n) = 0;
    virtual Status Close() = 0;

protected:
    PallocSequentialFile() = default;

    // Aligned buffer for reads (4096-byte aligned for Direct I/O)
    void* read_buffer_{nullptr};
    size_t buffer_capacity_{0};
    size_t buffer_pos_{0};
    size_t buffer_valid_{0};
};

// =============================================================================
// PallocRandomAccessFile — Position-based read-only access
// =============================================================================

class PallocRandomAccessFile {
public:
    // Open file for random access reading
    static Status Open(const char* path, alloc::UniquePtr<PallocRandomAccessFile>* out);

    virtual ~PallocRandomAccessFile();

    // Read n bytes at offset. Result points to internal palloc buffer.
    // Buffer valid until next Read() or Close().
    virtual Status Read(uint64_t offset, size_t n, Slice* result) = 0;
    virtual Status Close() = 0;

protected:
    PallocRandomAccessFile() = default;

    // Aligned buffer for reads (4096-byte aligned for Direct I/O)
    void* read_buffer_{nullptr};
    size_t buffer_capacity_{0};
};

// =============================================================================
// PallocWritableFile — Append-only write with palloc buffers
// =============================================================================

class PallocWritableFile {
public:
    // Create new file for writing
    static Status Create(const char* path, alloc::UniquePtr<PallocWritableFile>* out);
    // Open existing file for append (WAL use case)
    static Status OpenAppend(const char* path, alloc::UniquePtr<PallocWritableFile>* out);

    virtual ~PallocWritableFile();

    // Append data to file. Data may be copied to internal palloc buffer.
    virtual Status Append(Slice data) = 0;
    virtual Status Pwrite(uint64_t offset, Slice data) = 0;
    virtual uint64_t BytesWritten() const = 0;
    virtual Status Flush() = 0;
    virtual Status Sync() = 0;
    virtual Status Close() = 0;

protected:
    PallocWritableFile() = default;

    // Aligned buffer for writes (4096-byte aligned for Direct I/O)
    void* write_buffer_{nullptr};
    size_t buffer_capacity_{0};
    size_t buffer_pos_{0};
    uint64_t bytes_written_{0};
};

// =============================================================================
// PallocFileMapping — Read-only view with palloc backing
// =============================================================================

class PallocFileMapping {
public:
    // Map file into memory using palloc backing
    static Status Map(const char* path, alloc::UniquePtr<PallocFileMapping>* out);

    virtual ~PallocFileMapping();

    virtual const void* Data() const = 0;
    virtual size_t Size() const = 0;

protected:
    PallocFileMapping() = default;

    void* mapped_data_{nullptr};
    size_t mapped_size_{0};
};

// =============================================================================
// Filesystem operations
// =============================================================================

class PallocFilesystem {
public:
    // Check if file exists
    static Status FileExists(const char* path);

    // Get file size
    static Status GetFileSize(const char* path, uint64_t* size);

    // Delete file (renamed to avoid Windows macro conflict)
    static Status RemoveFile(const char* path);

    // Delete directory recursively
    static Status RemoveDirRecursive(const char* path);

    // Create directory (recursively if needed)
    static Status CreateDir(const char* path);

    // Sync directory metadata
    static Status SyncDir(const char* path);
};

// =============================================================================
// Alignment constants for Direct I/O compatibility
// =============================================================================

constexpr size_t kIoAlignment = 4096;  // OS page / disk sector alignment
constexpr size_t kDefaultIoBufferSize = 64 * 1024;  // 64KB I/O buffer

} // namespace pomai::storage
