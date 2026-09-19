// include/pio/pio_platform.h — Platform detection and hardware alignment
// Copyright 2026 PomaiDB / pio authors. MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if defined(_WIN32) || defined(_WIN64)
    #define PIO_PLATFORM_WINDOWS 1
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #define PIO_PLATFORM_POSIX 1
    #if defined(__APPLE__)
        #define PIO_PLATFORM_DARWIN 1
    #elif defined(__linux__)
        #define PIO_PLATFORM_LINUX 1
    #endif
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

namespace pio {

/// Sector size for Direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING)
constexpr std::size_t kDirectIoSectorSize = 4096;
constexpr std::size_t kSectorSize = kDirectIoSectorSize;

/// Default OS memory page size
constexpr std::size_t kPageSize = 4096;

/// Default streaming read chunk size (1MB)
constexpr std::size_t kStreamReadChunkSize = 1024 * 1024;

/// Default write buffer capacity (64KB)
constexpr std::size_t kDefaultWriteBufferSize = 65536;

inline void* direct_alloc(std::size_t bytes, std::size_t alignment = kDirectIoSectorSize) noexcept {
    if (bytes == 0) return nullptr;
#if defined(PIO_PLATFORM_WINDOWS)
    return _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, alignment, bytes) != 0) return nullptr;
    return ptr;
#endif
}

inline void direct_free(void* ptr) noexcept {
    if (!ptr) return;
#if defined(PIO_PLATFORM_WINDOWS)
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

} // namespace pio
