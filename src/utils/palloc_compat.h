// PomaiDB memory shim: routes aligned heap allocations through vendored palloc
// (pa_malloc_aligned / pa_free / pa_heap_*). Requires linking palloc-static.
#pragma once

#ifndef PA_VECTOR
#define PA_VECTOR 1
#endif

#include <palloc.h>
#include <palloc_vector.h>

#ifdef __cplusplus
#include <cstddef>
#include <limits>
#include <new>

typedef pa_heap_t palloc_heap_t;

namespace pomai::util {
bool EnsurePallocInitialized();
}

inline bool palloc_is_owned(const void* p) {
  if (!p) return false;
  return pa_is_in_heap_region(p) || pa_usable_size(p) > 0;
}

namespace pomai {

template <typename T, std::size_t Alignment = 64>
class PallocAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = PallocAllocator<U, Alignment>;
    };

    constexpr PallocAllocator() noexcept = default;
    template <typename U>
    constexpr PallocAllocator(const PallocAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        std::size_t align = (Alignment < alignof(T)) ? alignof(T) : Alignment;
        void* p = pa_malloc_aligned(n * sizeof(T), align);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        (void)n;
        if (p) pa_free(p);
    }

    template <typename U, std::size_t A>
    bool operator==(const PallocAllocator<U, A>&) const noexcept { return Alignment == A; }
    template <typename U, std::size_t A>
    bool operator!=(const PallocAllocator<U, A>&) const noexcept { return Alignment != A; }
};

} // namespace pomai

inline void* palloc_malloc_aligned(std::size_t size, std::size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return pa_malloc_aligned(size, alignment);
}

inline void palloc_free(void* p) { pa_free(p); }

inline palloc_heap_t* palloc_heap_new(void) { return pa_heap_new(); }

inline void palloc_heap_delete(palloc_heap_t* heap) {
  if (heap) {
    pa_heap_delete(heap);
  }
}

inline void* palloc_heap_malloc_aligned(palloc_heap_t* heap, std::size_t size,
                                        std::size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return heap ? pa_heap_malloc_aligned(heap, size, alignment)
              : pa_malloc_aligned(size, alignment);
}

inline void palloc_option_set(long option, long value) {
  pa_option_set(static_cast<pa_option_t>(option), value);
}

static constexpr pa_option_t palloc_option_reserve_huge_os_pages =
    pa_option_reserve_huge_os_pages;

#else

#include <stddef.h>

typedef pa_heap_t palloc_heap_t;

static inline void* palloc_malloc_aligned(size_t size, size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return pa_malloc_aligned(size, alignment);
}

static inline void palloc_free(void* p) { pa_free(p); }

static inline palloc_heap_t* palloc_heap_new(void) { return pa_heap_new(); }

static inline void palloc_heap_delete(palloc_heap_t* heap) {
  if (heap) {
    pa_heap_delete(heap);
  }
}

static inline void* palloc_heap_malloc_aligned(palloc_heap_t* heap, size_t size,
                                               size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return heap ? pa_heap_malloc_aligned(heap, size, alignment)
              : pa_malloc_aligned(size, alignment);
}

static inline void palloc_option_set(long option, long value) {
  pa_option_set((pa_option_t)option, value);
}

#define palloc_option_reserve_huge_os_pages pa_option_reserve_huge_os_pages

#endif
