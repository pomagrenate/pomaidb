# PomaiDB palloc — ATTACK SURFACE ANALYSIS & ARCHITECTURE FORENSICS

Generated: 2026-09-13
Subsystem: palloc (vendored high-performance memory allocator & PomaiDB extensions)

---

## 1. Executive Summary & Component Breakdown

PomaiDB relies on **palloc**, an integrated, hardened memory subsystem based on a specialized mimalloc derivative (	hird_party/palloc), extended with custom vector allocators, SIMD bump arenas, fixed-size object pools, thread-safe shims, and a virtual swap-backed page pool (palloc_page_pool).

### Architecture Layers:
1. **Core Allocator (	hird_party/palloc)**:
   - Multi-tiered size classes (73 bins, exponential 12.5% progression, O(1) table lookup for sizes <= 8192 bytes).
   - Multi-tiered free lists: Thread-local ree, local deferred local_free, cross-thread atomic lock-free queue 	hread_free.
   - Thread-local heaps (pa_heap_t) managing small/medium/large/huge pages.
   - Slices (64 KiB), Segments (32 MiB on 64-bit), Arenas (OS level virtual memory commit/decommit, NUMA-aware, huge page management).
   - Segment map: Radix-tree bitset mapping OS addresses to segment cookies.
2. **Pomai Vector Module (	hird_party/palloc/src/palloc_vector.c)**:
   - pa_vec_arena_t: Dynamic chunk-doubling bump arena for contiguous SIMD/vector buffers.
   - pa_vec_pool_t: Fixed-size object pool with embedded linked-list free chains.
   - pa_vector_batch_alloc, pa_vector_batch_free, pa_vector_batch_alloc_floats: Batch vector allocation primitives.
3. **Pomai Arena Module (	hird_party/palloc/src/arena_pomai.c)**:
   - p_arena_vector_t: Direct OS-mapped (_pa_prim_alloc) contiguous vector arena.
   - Dual-mode operation: Plain single-threaded bump pointer vs Shared multi-threaded atomic bump pointer (offset_atomic).
   - Hard fuse (max_capacity_bytes) for zero-OOM guarantees (returns NULL when full, never asks OS or aborts).
   - Scavenger reset (p_arena_reset): Decommits physical memory while retaining virtual address space.
4. **PomaiDB Integration Shims (src/utils/palloc_*)**:
   - src/utils/palloc_override.cc: Replaces global ISO C++ operator new, delete, sized delete, and C++17 aligned new/delete.
   - src/utils/palloc_compat.h: PallocAllocator<T, Alignment> C++ STL allocator, palloc_malloc_aligned, palloc_free, palloc_is_owned.
   - src/utils/palloc_page_pool.cc: C-compatible page cache with CLOCK eviction, dirty page tracking, and Win32/POSIX file swap backing.

---

## 2. Forensic Mapping of Attack Surfaces

### 2.1 Core Allocation & Deallocation Entry Points
- pa_malloc(size_t size)
- pa_calloc(size_t count, size_t size)
- pa_realloc(void* p, size_t newsize)
- pa_free(void* p)
- pa_free_size(void* p, size_t size)
- pa_free_aligned(void* p, size_t alignment)
- pa_free_size_aligned(void* p, size_t size, size_t alignment)
- pa_malloc_aligned(size_t size, size_t alignment)
- pa_realloc_aligned(void* p, size_t newsize, size_t alignment)

**Attack Vectors**:
- Boundary size inputs: $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $ (small/medium boundary), $, $, $ (medium/large boundary), \text{ MiB}$ (large/huge boundary).
- Integer overflow: size > SIZE_MAX - alignment, count * size > SIZE_MAX, SIZE_MAX, SIZE_MAX - 1, PTRDIFF_MAX.
- Alignment: Non-power-of-two alignments (, 3, 5, 7, 13$), zero alignment, massive alignments (\text{ KiB}, 1\text{ MiB}, 32\text{ MiB}$).
- Corrupted pointers: 
ullptr, interior pointers, stack pointers, foreign pointers (system malloc), double free, use-after-free.

### 2.2 Vector Module (palloc_vector.c)
- pa_vec_pool_create(size_t object_size, size_t initial_count)
- pa_vec_pool_alloc(pa_vec_pool_t* pool)
- pa_vec_pool_free(pa_vec_pool_t* pool, void* ptr)
- pa_vec_pool_destroy(pa_vec_pool_t* pool)
- pa_vec_arena_create(size_t initial_size, size_t alignment)
- pa_vec_arena_alloc(pa_vec_arena_t* arena, size_t size)
- pa_vec_arena_reset(pa_vec_arena_t* arena)
- pa_vec_arena_destroy(pa_vec_arena_t* arena)
- pa_vector_batch_alloc(size_t count, size_t size)
- pa_vector_batch_free(void** ptrs, size_t count)
- pa_vector_batch_alloc_floats(size_t count, size_t dim)

**Attack Vectors**:
- pa_vec_pool: object_size < sizeof(void*) (sub-pointer size overwrite), page-chain severance bug on *(void**)page overwrite, arbitrary address free on destroy, unaligned pointer access.
- pa_vec_arena: Doubling loop integer overflow hang on ligned_size > SIZE_MAX / 2, chunk header alignment overflow.
- pa_vector_batch_alloc_floats: dim * sizeof(float) overflow wrapping around to allocate tiny buffer.
- Partial batch allocation failure cleanup.

### 2.3 Vector Arena (rena_pomai.c)
- p_arena_create_for_vector_ex(size_t capacity_bytes, bool shared)
- p_arena_alloc_vector(pa_arena_t* arena, size_t vector_dim, size_t element_size)
- p_arena_reset(pa_arena_t* arena)
- p_arena_destroy(pa_arena_t* arena)

**Attack Vectors**:
- Atomic rollback race condition in shared=true: Thread A pushes offset past capacity, Thread B arrives concurrently, both rollback, non-linearizable allocation state or false OOM.
- Integer overflow in old + aligned_size: Bypasses hard_limit check if old + aligned_size wraps around zero.
- Data race on 
eeds_commit_after_reset across multiple threads during recommit after scavenger decommit.

### 2.4 Page Pool (palloc_page_pool.cc)
- palloc_page_pool_create(const palloc_page_pool_options* opts)
- palloc_fetch_page(palloc_page_pool* pool, uint64_t page_id, int for_write, int* is_new)
- palloc_unpin_page(palloc_page_pool* pool, uint64_t page_id, int mark_dirty)
- palloc_flush_page(palloc_page_pool* pool, uint64_t page_id)
- palloc_flush_all(palloc_page_pool* pool)
- palloc_page_pool_destroy(palloc_page_pool* pool)
- palloc_get_default_page_pool(const char* swap_file_path_hint)

**Attack Vectors**:
- Complete lack of concurrency synchronization: Concurrent palloc_fetch_page / palloc_unpin_page corrupting std::unordered_map and std::vector internal states.
- Integer overflow in max_resident_pages * page_size.
- Capacity limits: $ page, $ capacity, capacity smaller than page size.
- Page ID edge cases: kInvalidPageId (^{64}-1$), colliding with internal unassigned sentinels.
- Swap I/O failure handling: Read/write failures, disk full, missing swap path.

### 2.5 Ownership & Shims (palloc_compat.h, palloc_override.cc)
- palloc_is_owned(const void* p): Falling through to pa_usable_size on non-heap pointers.
- Global C++ replacement operator new/delete: Sized deallocation mismatches, aligned deallocation mismatches.
- STL container reallocations and allocator rebinding.
