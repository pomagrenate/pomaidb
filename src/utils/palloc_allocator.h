// palloc_allocator.h — Scoped C++17 STL-compliant allocator wrapper around palloc
//
// This provides a safe, scoped alternative to global operator new/delete override.
// Use this allocator explicitly for internal containers that benefit from arena allocation.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <new>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <palloc.h>
#include "palloc_compat.h"

namespace pomai::alloc {

// constexpr alignment for SIMD operations (AVX2 = 32-byte, AVX-512 = 64-byte)
constexpr std::size_t kSimdAlignment = 64;

// PallocAllocator: STL-compliant allocator using palloc
template <typename T>
class PallocAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template <typename U>
    struct rebind {
        using other = PallocAllocator<U>;
    };

    PallocAllocator() noexcept = default;
    
    template <typename U>
    PallocAllocator(const PallocAllocator<U>&) noexcept {}
    
    T* allocate(std::size_t n) {
        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        // Use aligned allocation for SIMD-friendly memory
        // Ensure minimum 64-byte alignment for AVX-512
        void* p = pa_malloc_aligned(n * sizeof(T), kSimdAlignment);
        if (!p) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }
    
    // Allocate with explicit alignment (for special cases)
    T* allocate_aligned(std::size_t n, std::size_t alignment) {
        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        void* p = pa_malloc_aligned(n * sizeof(T), alignment);
        if (!p) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }
    
    void deallocate(T* p, std::size_t n) noexcept {
        (void)n; // Unused but required by STL allocator interface
        if (p) {
            pa_free_aligned(p, kSimdAlignment);
        }
    }
    
    // Equality operators (stateless allocators are always equal)
    bool operator==(const PallocAllocator&) const noexcept { return true; }
    bool operator!=(const PallocAllocator&) const noexcept { return false; }
};

// Specialization for void (required by STL)
template <>
class PallocAllocator<void> {
public:
    using value_type = void;
    using pointer = void*;
    using const_pointer = const void*;
    
    template <typename U>
    struct rebind {
        using other = PallocAllocator<U>;
    };
};

// Type aliases for common container usage
template <typename T>
using PallocVector = std::vector<T, PallocAllocator<T>>;

template <typename K, typename V, typename Hash = std::hash<K>>
using PallocUnorderedMap = std::unordered_map<K, V, Hash, std::equal_to<K>, PallocAllocator<std::pair<const K, V>>>;

// Helper function to create palloc-allocated containers
template <typename T, typename... Args>
PallocVector<T> MakePallocVector(Args&&... args) {
    return PallocVector<T>(std::forward<Args>(args)...);
}

} // namespace pomai::alloc