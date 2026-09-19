// include/pio/pio_file.h — Abstract interfaces for file access
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "pio_platform.h"
#include "pio_status.h"
#include "pio_types.h"

namespace pio {

/// SequentialFile: Streaming sequential read-only access (WAL replay, batch scans).
class SequentialFile {
public:
    virtual ~SequentialFile() = default;

    /// Read up to n bytes from file. Result points to data read (either in scratch or internal buffer).
    virtual Status Read(std::size_t n, Slice* result, char* scratch) = 0;

    /// Skip over n bytes from current position.
    virtual Status Skip(std::uint64_t n) = 0;

    /// Close the file.
    virtual Status Close() = 0;
};

/// RandomAccessFile: Thread-safe, position-based read-only access.
class RandomAccessFile {
public:
    virtual ~RandomAccessFile() = default;

    /// Read n bytes starting at offset. Multiple threads can call Read concurrently safely.
    virtual Status Read(std::uint64_t offset, std::size_t n, Slice* result, char* scratch) const = 0;

    /// Advise OS that data in range [offset, offset + n) will be needed soon.
    virtual void Prefetch(std::uint64_t offset, std::size_t n) const = 0;

    /// Close the file.
    virtual Status Close() = 0;
};

/// WritableFile: Append-only write access with user-space buffering and hardware sync gates.
class WritableFile {
public:
    virtual ~WritableFile() = default;

    /// Append data to file. May buffer in memory until Flush() or threshold.
    virtual Status Append(Slice data) = 0;

    /// Positional write (flushes pending append buffers first).
    virtual Status Pwrite(std::uint64_t offset, Slice data) = 0;

    /// Flush user-space buffer to OS page cache.
    virtual Status Flush() = 0;

    /// Sync OS page cache to physical non-volatile storage (fdatasync / FlushFileBuffers).
    virtual Status Sync() = 0;

    /// Flush pending writes and close the file handle.
    virtual Status Close() = 0;

    /// Returns total bytes written (including buffered bytes).
    [[nodiscard]] virtual std::uint64_t BytesWritten() const noexcept = 0;
};

/// MemoryMappedFile: RAII zero-copy read-only memory-mapped file.
class MemoryMappedFile {
public:
    virtual ~MemoryMappedFile() = default;

    [[nodiscard]] virtual const std::uint8_t* data() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;

    /// Issue madvise / PrefetchVirtualMemory hints
    virtual void Advise(Advice advice) = 0;

    /// Explicitly unmap and close underlying handle
    virtual void Close() = 0;
};

} // namespace pio
