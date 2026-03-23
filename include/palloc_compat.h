// Compatibility shim: provides palloc_* memory API using the system allocator.
// PomaiDB no longer depends on third_party/palloc; all aligned allocation goes
// through posix_memalign (Linux/macOS) or _aligned_malloc (Windows).
#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdlib>
#else
#include <stddef.h>
#include <stdlib.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef __cplusplus
typedef void* palloc_heap_t;

static inline void* palloc_malloc_aligned(std::size_t size, std::size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
#if defined(_WIN32) || defined(_WIN64)
  return _aligned_malloc(size, alignment);
#else
  void* p = nullptr;
  return (posix_memalign(&p, alignment, size) == 0) ? p : nullptr;
#endif
}

static inline void palloc_free(void* p) {
#if defined(_WIN32) || defined(_WIN64)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

static inline palloc_heap_t* palloc_heap_new(void) { return nullptr; }
static inline void palloc_heap_delete(palloc_heap_t*) {}
static inline void* palloc_heap_malloc_aligned(palloc_heap_t*, std::size_t size, std::size_t alignment) {
  return palloc_malloc_aligned(size, alignment);
}
static inline void palloc_option_set(long, long) {}
static constexpr long palloc_option_reserve_huge_os_pages = 0;
#else

typedef void* palloc_heap_t;

static inline void* palloc_malloc_aligned(size_t size, size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
#if defined(_WIN32) || defined(_WIN64)
  return _aligned_malloc(size, alignment);
#else
  void* p = NULL;
  return (posix_memalign(&p, alignment, size) == 0) ? p : NULL;
#endif
}

static inline void palloc_free(void* p) {
#if defined(_WIN32) || defined(_WIN64)
  _aligned_free(p);
#else
  free(p);
#endif
}

#define palloc_heap_new() (NULL)
#define palloc_heap_delete(x) ((void)0)
#define palloc_heap_malloc_aligned(h, s, a) palloc_malloc_aligned(s, a)
#define palloc_option_set(k, v) ((void)0)
#define palloc_option_reserve_huge_os_pages (0)
#endif
