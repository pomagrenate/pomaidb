#include <new>
#include <cstddef>
#include <cstdlib>
#include <palloc.h>
#include <palloc_vector.h>
#include "palloc_compat.h"

namespace pomai::util {

bool EnsurePallocInitialized() {
    static const bool initialized = []() {
        return pa_version() > 0;
    }();
    return initialized;
}

} // namespace pomai::util

// ============================================================================
// Global ISO C++ Replaceable Allocation Functions
// All allocations inside PomaiDB unconditionally route through palloc.
// ============================================================================

void operator delete(void* p) noexcept {
    pa_free(p);
}

void operator delete[](void* p) noexcept {
    pa_free(p);
}

void operator delete(void* p, const std::nothrow_t&) noexcept {
    pa_free(p);
}

void operator delete[](void* p, const std::nothrow_t&) noexcept {
    pa_free(p);
}

void* operator new(std::size_t n) noexcept(false) {
    return pa_new(n);
}

void* operator new[](std::size_t n) noexcept(false) {
    return pa_new(n);
}

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    return pa_new_nothrow(n);
}

void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    return pa_new_nothrow(n);
}

// C++14 sized delete
void operator delete(void* p, std::size_t n) noexcept {
    pa_free_size(p, n);
}

void operator delete[](void* p, std::size_t n) noexcept {
    pa_free_size(p, n);
}

// C++17 aligned allocation
void operator delete(void* p, std::align_val_t al) noexcept {
    pa_free_aligned(p, static_cast<std::size_t>(al));
}

void operator delete[](void* p, std::align_val_t al) noexcept {
    pa_free_aligned(p, static_cast<std::size_t>(al));
}

void operator delete(void* p, std::size_t n, std::align_val_t al) noexcept {
    pa_free_size_aligned(p, n, static_cast<std::size_t>(al));
}

void operator delete[](void* p, std::size_t n, std::align_val_t al) noexcept {
    pa_free_size_aligned(p, n, static_cast<std::size_t>(al));
}

void operator delete(void* p, std::align_val_t al, const std::nothrow_t&) noexcept {
    pa_free_aligned(p, static_cast<std::size_t>(al));
}

void operator delete[](void* p, std::align_val_t al, const std::nothrow_t&) noexcept {
    pa_free_aligned(p, static_cast<std::size_t>(al));
}

void* operator new(std::size_t n, std::align_val_t al) noexcept(false) {
    return pa_new_aligned(n, static_cast<std::size_t>(al));
}

void* operator new[](std::size_t n, std::align_val_t al) noexcept(false) {
    return pa_new_aligned(n, static_cast<std::size_t>(al));
}

void* operator new(std::size_t n, std::align_val_t al, const std::nothrow_t&) noexcept {
    return pa_new_aligned_nothrow(n, static_cast<std::size_t>(al));
}

void* operator new[](std::size_t n, std::align_val_t al, const std::nothrow_t&) noexcept {
    return pa_new_aligned_nothrow(n, static_cast<std::size_t>(al));
}
