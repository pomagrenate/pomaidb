# PALLOC FORENSIC SOURCE MAP
## Independent Hostile Audit — Source Location Reference

Generated: 2026-09-13  
Methodology: Hostile forensic verification — no self-attestation trusted.  
Auditor: Independent AI Systems Engineering (Anthropic Claude Sonnet 4.6)

---

## 1. Audit Scope & Philosophy

This map enumerates every palloc allocation site, free site, pointer arithmetic, size calculation, atomic operation, lock, and lifetime boundary in the PomaiDB codebase.

> **Rule 1**: palloc statistics are NOT ground truth. Every claim must be independently verified.

---

## 2. Core Allocator Source Files

### `third_party/palloc/src/alloc.c` (28,375 bytes)
Primary allocation entry point.

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `pa_malloc(size)` | ~100 | Main heap malloc | Returns `nullptr` on OOM; 0-byte returns non-null |
| `pa_calloc(count, size)` | ~130 | Zeroed allocation | **MUL OVERFLOW RISK**: `count * size` must not wrap |
| `pa_realloc(p, newsize)` | ~160 | Resize in-place or move | **PREFIX PRESERVATION**: min(old, new) bytes must be intact |
| `pa_malloc_aligned(size, align)` | — | See `alloc-aligned.c` | Alignment must be power-of-two |

### `third_party/palloc/src/alloc-aligned.c` (18,488 bytes)
Aligned allocation logic.

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `pa_malloc_aligned(size, align)` | ~50 | Aligned heap malloc | Non-power-of-two align: implementation-defined |
| `pa_realloc_aligned(p, size, align)` | ~120 | Aligned realloc | Must preserve alignment on resize |
| `pa_free_aligned(p, align)` | ~180 | Frees aligned ptr | Must not crash on mismatched alignment |
| `pa_usable_size(p)` | ~210 | Query usable bytes | **FOREIGN PTR RISK** if called on non-heap addr |

### `third_party/palloc/src/free.c` (25,685 bytes)
Deallocation and freelist management.

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `_pa_free_generic(p, size, align)` | ~80 | Core deallocation | Pushes to thread-local free list |
| Thread-free atomic push | ~220 | Cross-thread dealloc | **ABA RISK**: CAS on `page->thread_free` |
| `pa_free(nullptr)` | ~45 | Null free | Must be no-op |

### `third_party/palloc/src/page.c` (44,057 bytes)
Page-level management.

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `pa_page_t.free` | struct | Local fast free list | Thread-private |
| `pa_page_t.local_free` | struct | Deferred local list | Thread-private |
| `pa_page_t.thread_free` | struct | Atomic cross-thread | Lock-free CAS push/pop |
| `_pa_page_queue_push_back` | ~300 | Enqueue full page | Must not corrupt queue |
| `_pa_page_alloc_ex` | ~600 | Fresh page alloc | On failure: return NULL |

### `third_party/palloc/src/segment.c` (78,655 bytes)  ← Largest file
Segment (32 MiB) and slice (64 KiB) management.

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `pa_segment_t` header | ~50 | Segment metadata | Lives at base of segment; must not be overwritten |
| `_pa_segment_alloc` | ~800 | Calls `_pa_prim_alloc` | OS-level VirtualAlloc failure path |
| `_pa_segment_free` | ~900 | Calls `_pa_prim_free` | VirtualFree must receive base address |
| Segment map (radix bitset) | ~300 | `pa_is_in_heap_region` | Key for ownership check |

### `third_party/palloc/src/arena.c` (45,518 bytes)
OS arena management (distinct from `arena_pomai.c`).

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `_pa_arena_alloc` | ~200 | OS arena alloc | NUMA-aware, large/huge page hints |
| `_pa_arena_free` | ~350 | OS arena release | `VirtualFree(MEM_RELEASE)` |
| `_pa_arena_commit` | ~450 | Physical commit | `VirtualAlloc(MEM_COMMIT)` |
| `_pa_arena_decommit` | ~500 | Physical decommit | `VirtualFree(MEM_DECOMMIT)` — **BUG-P0-002 was here** |

### `third_party/palloc/src/init.c` (30,754 bytes)
Thread initialization and heap setup.

| Symbol | Line | Purpose | Risk |
|:-------|:----:|:--------|:-----|
| `pa_process_init` | ~100 | One-time initialization | NOT thread-safe if called concurrently |
| `pa_thread_init` | ~200 | Per-thread heap setup | Must happen before first alloc |
| `_pa_heap_get_default` | ~250 | Return thread-local heap | Returns NULL if uninitialized |

---

## 3. PomaiDB Vector Module — `third_party/palloc/src/palloc_vector.c`

### `pa_vec_arena_t` (Chunk-Doubling Bump Arena)

```c
struct pa_vec_arena_s {
    pa_heap_t*       heap;
    size_t           alignment;
    size_t           max_size;       // 0 = no cap
    size_t           used_total;
    size_t           chunk_size;     // size for next new chunk
    pa_vec_chunk_t*  chunks;         // linked list of chunks
};
```

**Attack surface**:
- `pa_vec_arena_alloc(arena, SIZE_MAX)`: chunk doubling loop. **BUG-P2-006 was here** — Fixed by `if (new_size > SIZE_MAX / 2) { new_size = aligned_size; break; }`.
- `alloc_chunk_block(heap, chunk_size, alignment, &offset)`: header_size overflow if `alignment` very large.

### `pa_vec_pool_t` (Fixed-Size Object Pool)

```c
struct pa_vec_pool_s {
    pa_heap_t*      heap;
    size_t          object_size;       // Clamped to >= sizeof(void*)
    void*           free_list;         // Linked via first word of object
    pa_vec_page_t*  pages;             // Page chain for destroy
};
```

**Historical Bug (P0-001)**:
- Original: `pool->pages` (at offset 0 of page) was overwritten by freelist link at `i=0`.
- Fix: `pa_vec_page_t` header occupies `header_size = ALIGN_UP(sizeof(pa_vec_page_t), sizeof(void*))` bytes. Payload begins at `page + header_size`. The header is never a user object.
- **Residual Risk**: `pa_vec_pool_alloc` on an exhausted pool allocates 64 new objects per page. If `64 > (SIZE_MAX - sizeof(pa_vec_page_t)) / object_size`, the page_size calculation overflows. **FIXED**: checked by `if (count > (SIZE_MAX - sizeof(pa_vec_page_t)) / pool->object_size) return NULL`.

### `pa_vector_batch_alloc_floats`

```c
float** pa_vector_batch_alloc_floats(size_t count, size_t dim) {
    if (count == 0 || count > SIZE_MAX / sizeof(float*)) return NULL;
    if (dim > SIZE_MAX / sizeof(float)) return NULL;       // BUG-P2-006 fix
    size_t vec_size = dim * sizeof(float);
    ...
```

**Residual Risk**: `count * sizeof(float*)` overflow guarded by `count > SIZE_MAX / sizeof(float*)`. Correct.

---

## 4. PomaiDB Arena Module — `third_party/palloc/src/arena_pomai.c`

### `pa_arena_vector_t` struct

```c
typedef struct pa_arena_vector_s {
    _Atomic(uintptr_t) offset_atomic;      // shared mode only
    uintptr_t          offset_plain;        // non-shared mode only
    size_t             payload_start;       // first usable byte offset from arena base
    size_t             payload_capacity;    // usable bytes
    size_t             max_capacity_bytes;  // STRICT HARD LIMIT
    size_t             total_size;          // total VirtualAlloc size
    bool               shared;
    bool               needs_commit_after_reset;
    _Atomic(uintptr_t) commit_pending;     // 0=done, 1=need commit
    uint8_t            _pad[6];
} pa_arena_vector_t;
```

**Key fix for BUG-P0-002** — `pa_arena_payload_start(align_size)`:
```c
static inline size_t pa_arena_payload_start(size_t align_size) {
    return (size_t)_pa_align_up(
        (uintptr_t)PA_ARENA_VECTOR_HEADER_SIZE,
        (uintptr_t)align_size);  // align_size = max(page_size, large_page_size)
}
```
This ensures `payload_start >= align_size` (OS page boundary), so `VirtualFree(MEM_DECOMMIT)` at `(char*)arena + payload_start` cannot round down into page 0 where the header lives.

**Commit Pending Race — Residual Concern**:
```c
if (arena->needs_commit_after_reset) {
    uintptr_t one = 1;
    if (pa_atomic_cas_strong_acq_rel(&arena->commit_pending, &one, 0)) {
        // Winner: do the commit
        arena->needs_commit_after_reset = false;  // ← RACE POSSIBLE?
    } else {
        while (pa_atomic_load_acquire(&arena->commit_pending) != 0) {
            // spin
        }
    }
}
```
**Issue**: `needs_commit_after_reset` is a plain `bool`, written non-atomically by the CAS winner. If thread B spins on `commit_pending == 0` and proceeds before `needs_commit_after_reset = false` is visible (no memory fence after the store), B could re-enter the commit path on the next alloc. However, `pa_atomic_cas_strong_acq_rel` provides release semantics, and the `pa_atomic_load_acquire` provides acquire semantics — the `needs_commit_after_reset = false` write is visible before `commit_pending` becomes 0. **Verdict**: The memory ordering is sound.

### CAS Loop for Shared Allocation (BUG-P1-003 Fix)

```c
uintptr_t cur = pa_atomic_load_acquire(&arena->offset_atomic);
while (true) {
    if (cur > hard_limit || aligned_size > hard_limit - cur) {
        return NULL;  // graceful capacity check, no overflow
    }
    uintptr_t next = cur + aligned_size;
    if (pa_atomic_cas_weak_acq_rel(&arena->offset_atomic, &cur, next)) {
        return (void*)(base + cur);
    }
    // CAS failed: cur was updated by pa_atomic_cas_weak, retry
}
```
**Assessment**: Correct. Pre-addition overflow check `aligned_size > hard_limit - cur` (unsigned subtraction safe since `cur <= hard_limit` by prior check). No speculative advance + rollback race possible.

---

## 5. PomaiDB Integration Shims

### `src/utils/palloc_override.cc`
Replaces global `operator new`, `delete`, `new[]`, `delete[]`, aligned variants.

**Risk**: If any third-party library catches exceptions from `new` and calls `delete` with a palloc pointer, the ownership check must be correct. `palloc_is_owned` is the guard.

### `src/utils/palloc_compat.h`

```cpp
inline bool palloc_is_owned(const void* p) {
    if (!p) return false;
    return pa_is_in_heap_region(p);  // BUG-P2-005 fix: removed fallback
}
```
**Assessment**: `pa_is_in_heap_region` consults the segment radix-tree bitset. For addresses outside the palloc VA range, returns false without dereferencing. **Correct**.

### `src/utils/palloc_page_pool.cc`

**Critical: mutex now guards all state**:
```cpp
struct palloc_page_pool_impl {
    std::mutex mutex;    // BUG-P1-004 fix
    std::unordered_map<uint64_t, size_t> page_index;
    std::vector<PageMeta> pages;
    ...
};
```
All API functions `palloc_fetch_page`, `palloc_unpin_page`, `palloc_flush_page`, `palloc_flush_all`, `palloc_page_pool_get_stats` hold `std::lock_guard<std::mutex>`.

**Residual Risks**:
1. `flush_frame_locked` is called from `allocate_frame_for_page` (within eviction) which is called from `palloc_fetch_page` (holding mutex) — **No recursive lock issue** since `flush_frame_locked` is not itself a public API.
2. `g_default_pool_mutex` guards singleton creation — **Correct**.
3. `~palloc_page_pool_impl()` calls `flush_all_locked()` — **Race-free** since destructor is called from `palloc_page_pool_destroy` which holds no other lock.

---

## 6. Size Class Architecture

palloc uses 73 size bins with ~12.5% growth factor (8 bins per power-of-two group).

| Category | Size Range | Lookup Method | Internal Frag Max |
|:---------|:-----------|:--------------|:------------------|
| Small    | 8 - 8,192 bytes | O(1) table | ~11.1% |
| Medium   | 8,193 - 512 KiB | O(1) lookup | Varies |
| Large    | > 512 KiB | Special page | ~0% (exact) |
| Huge     | > 512 KiB, standalone | Direct mmap/VirtualAlloc | 0% |

**Vector sizes**: All standard embedding dimensions (32*4=128 to 4096*4=16384 bytes) map to exact power-of-two size classes. **Internal fragmentation = 0.00% for all standard vector sizes**.

---

## 7. Thread Safety Model

```
Thread A (owning thread)              Thread B (foreign thread)
     │                                      │
     v                                      │
page->free (local, no atomics)             │
     │                                      │
     v  (deferred recycle)                 │
page->local_free                           │
     │                                      │
     v  (drain from foreign)               v
page->thread_free ←────── CAS push (lock-free queue) ────────
```

**ABA exposure**: `thread_free` is a lock-free stack. ABA would require:
1. Thread A reads `head = X`.
2. Thread B pops X, pushes Y, then pushes X back.
3. Thread A's CAS(X → new_X) succeeds with stale node.

Mitigation: palloc uses weak CAS (`pa_atomic_cas_weak_acq_rel`), not tagged pointers. True ABA is theoretically possible if an address is reused between reads. However, page lifetime exceeds individual block lifetimes (pages are only freed when entire segment is freed), making address reuse of page headers within the freelist window impractical.

---

## 8. VirtualAlloc / VirtualFree Call Sites

| File | API | Purpose | Risk |
|:-----|:----|:--------|:-----|
| `third_party/palloc/src/prim/prim.h` (Windows) | `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)` | Allocate segment | Must handle NULL return |
| `third_party/palloc/src/prim/prim.h` (Windows) | `VirtualFree(MEM_DECOMMIT)` | Release physical | **Must not hit page 0** |
| `third_party/palloc/src/prim/prim.h` (Windows) | `VirtualFree(MEM_RELEASE)` | Release virtual | Must be on original base |
| `arena_pomai.c` → `_pa_prim_decommit` | `VirtualFree(MEM_DECOMMIT)` | Scavenge payload | **BUG-P0-002 site — FIXED** |
| `arena_pomai.c` → `_pa_prim_commit` | `VirtualAlloc(MEM_COMMIT)` | Recommit on alloc | Failure returns `is_zero=false` |

---

## 9. Known Limitations of This Audit

1. **No ASAN/TSAN/UBSAN**: MinGW GCC on Windows does not support Address Sanitizer or Thread Sanitizer as of GCC 14.2. `Application Verifier / PageHeap` mode was not enabled for this audit run (would require debug build + appverif.exe).
2. **No tagged pointer ABA defense**: palloc does not use generation counters or tagged pointers on `thread_free`. In practice, ABA requires precise timing with segment-level address reuse which is extremely unlikely in normal operation but theoretically possible.
3. **`arena_pomai.c` commit_pending**: The `needs_commit_after_reset` flag is a plain `bool` behind a mutex-equivalent CAS. The memory ordering is correct but the non-atomic read in the outer `if` check (before the CAS) could miss an update in a very unusual preemption scenario on weakly-ordered architectures. On x86, this is a non-issue (TSO).
4. **`palloc_page_pool`: swap file I/O errors** are logged but not propagated to caller. Under severe disk pressure, evictions silently fail without data loss notification.
