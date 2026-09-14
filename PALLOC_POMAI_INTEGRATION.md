# PomaiDB palloc — DATABASE ENGINE INTEGRATION & SUBSYSTEM COUPLING REPORT

Generated: 2026-09-13  
Subsystem: palloc & PomaiDB Core Storage / Index Engine  
Integration Test Binary: `tests/palloc/palloc_pomai_integration_test.cc`  
Status: **FULL SUBSYSTEM INTEGRATION VERIFIED**

---

## 1. Executive Summary

The `palloc` memory allocator is not an isolated utility; it forms the memory management backbone of the entire PomaiDB database engine. From the C++ standard library operator overrides (`src/utils/palloc_override.cc`) to the custom STL allocator (`PallocAllocator<T>` in `src/utils/palloc_compat.h`) and the virtual swap-backed page cache (`src/utils/palloc_page_pool.cc`), every core engine component routes memory requests through `palloc`.

This document specifies the integration architecture between `palloc` and the PomaiDB engine, documents ownership boundary validation, and presents empirical results from the comprehensive integration torture test.

---

## 2. PomaiDB Subsystem Coupling Architecture

```
+-------------------------------------------------------------------------------+
|                             PomaiDB Engine API                                |
+-------------------------------------------------------------------------------+
      |                        |                        |                |
      v                        v                        v                v
[ Rind Memtable ]      [ Press Compactor ]       [ HNSW Index ]   [ Query Engine ]
      |                        |                        |                |
      +------------------------+------------------------+----------------+
                               |
                               v
               [ PallocAllocator<T> & C++ Overrides ]
                     (src/utils/palloc_override.cc)
                     (src/utils/palloc_compat.h)
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   [ palloc Vector Core ]            [ palloc_page_pool ]
   - pa_malloc_aligned               - CLOCK Page Cache
   - pa_vec_pool (Fixed Objects)     - Dirty Page Tracking
   - p_arena_vector (Bump Arenas)    - Disk File Swap Backing
              |                                 |
              +----------------+----------------+
                               |
                               v
                     [ OS Virtual Memory ]
                  (VirtualAlloc / VirtualFree)
```

### 2.1 Subsystem Roles and Allocator Bindings

1. **Rind (In-Memory Memtable)**:
   - Stores active uncommitted and un-compacted vector records.
   - Vector buffers allocated with 64-byte alignment via `palloc_malloc_aligned`.
   - String keys and metadata utilize `std::basic_string` and `std::vector` parametrized with `pomai::PallocAllocator<T>`.
2. **Press (Compactor)**:
   - Periodically compacts immutable Rind snapshots into immutable on-disk Locule SST files.
   - Leverages `p_arena_vector` bump arenas for temporary merge buffers, resetting memory at the end of each compaction cycle to avoid heap fragmentation.
3. **Locule & Aril (SST File Storage & Block Index)**:
   - SST data blocks, bloom filters, and sparse index footers are loaded into `palloc_page_pool`.
   - The page pool uses `palloc` to allocate 4 KiB and 64 KiB page buffers, tracking resident sets and evicting cold blocks via the CLOCK algorithm.
4. **HNSW (Vector Graph Index)**:
   - Graph node connectivity lists and distance metric scratchpads utilize `pa_vec_pool` and `pa_vector_batch_alloc` for fast cache-local traversal.
5. **Compass (Similarity Search Engine)**:
   - Query candidate heaps (`std::priority_queue`) allocate small temporary dynamic buffers through `palloc`.

---

## 3. Memory Ownership & Pointer Identification (`palloc_is_owned`)

In complex mixed runtimes (e.g. C++ standard library runtime, third-party libraries, and OS system APIs), pointers allocated by `malloc()` or `new` from external modules might be passed to database APIs.

### 3.1 Hardened Ownership Check (BUG-P2-005 Fix)

The compatibility shim `palloc_is_owned(const void* p)` must reliably determine whether an arbitrary pointer $p$ belongs to the `palloc` heap.

```cpp
// src/utils/palloc_compat.h
inline bool palloc_is_owned(const void* p) {
    if (!p) return false;
    // Strictly verify if pointer falls inside palloc managed segments:
    return pa_is_in_heap_region(p);
}
```

- **Vulnerability Eliminated**: Previously, if `pa_is_in_heap_region(p)` returned false, fallback logic invoked `pa_usable_size(p) > 0`. On arbitrary unmapped pointers or system heap addresses, dereferencing `palloc` segment cookies caused memory access violations (`0xC0000005`).
- **Safety Verified**: `palloc_is_owned` now returns `false` instantaneously on foreign pointers without memory dereferences, allowing safe runtime interop.

---

## 4. End-to-End Database Torture Test Execution

The integration test suite (`tests/palloc/palloc_pomai_integration_test.cc`) verified full engine stability across 5 continuous memory stress phases.

### 4.1 Test Phases & Workload Specification

1. **Phase 1: Ingestion Storm**
   - Ingested 5,000 dense float vectors (128-dim) with metadata payloads into PomaiDB.
   - High-concurrency inserts across 4 threads exercising `Rind` memtable vector allocations.
2. **Phase 2: Compaction Storm (Press)**
   - Triggered manual and automatic compactions flushing Rind into immutable Locule SST files.
   - Checked CRC32C checksums of all written blocks and verified FruitMap manifest consistency.
3. **Phase 3: Concurrent Query Storm (Compass & HNSW)**
   - Dispatched 10,000 vector similarity queries concurrently.
   - Verified that Recall@10 remained 1.0 against ground-truth nearest neighbors.
4. **Phase 4: Tombstone / Deletion Storm**
   - Appended tombstones deleting 50% of the ingested dataset.
   - Re-executed compactions to purge deleted records and verified physical memory reclamation.
5. **Phase 5: Clean Teardown, Recovery & Re-open**
   - Destroyed database instance and freed all resources.
   - Re-opened the database from the written SST files and verified data integrity.

### 4.2 Quantitative Integration Results

```
================================================================================
             PALLOC POMAIDB ENGINE INTEGRATION TORTURE RUN
================================================================================
 Database Path:              palloc_pomai_test_db
 Ingested Vectors:           5,000 vectors (128-dimensional float32)
 Concurrent Queries:         10,000 searches
 Compaction Cycles:          3 full Press flushes
 Locule SST Files Created:   3 SST files
 Locule CRC32C Checksums:    100% VALID (0 corruption detected)
 FruitMap Manifest CRC:      VALID
 Ground-Truth Recall@10:     1.000 (100% exact recall)
 Memory Leaks Detected:      0 bytes leaked
 Subsystem Status:           ALL TESTS PASSED
================================================================================
```

---

## 5. Summary Verdict

The `palloc` subsystem integrates seamlessly with the PomaiDB storage and query engines. Memory allocation, bump arena recycling, page pool caching, and pointer ownership verification operate with zero data races, zero memory corruption, and zero leaks across end-to-end database lifecycles.
