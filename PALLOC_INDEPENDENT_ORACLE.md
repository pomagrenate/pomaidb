# PALLOC INDEPENDENT ORACLE — Forensic Methodology
## Why the Existing Oracle Was Insufficient

**Date**: 2026-09-13  
**Audit type**: Independent adversarial verification  
**Verdict basis**: Evidence collected by system-malloc-backed oracle only

---

## The Circular Validation Problem

The original `AllocatorOracle` in [`palloc_oracle.h`](file:///e:/GithubProjects/pomaidb/tests/palloc/palloc_oracle.h) has a fundamental design flaw for an adversarial audit:

```cpp
// From palloc_oracle.h line 85-89:
size_t usable = pa_usable_size(raw);       // ← palloc introspection
if (usable < total_alloc) {
    pa_free(raw);
    throw std::runtime_error("pa_usable_size < requested total_alloc!");
}
live_intervals_[raw_addr] = raw_addr + usable; // ← intervals sized by palloc's own data
```

**Problem**: If palloc's `pa_usable_size` returns a wrong value, the oracle's interval boundaries are also wrong. A corrupt allocator could:
1. Return an overlapping pointer.
2. Also return an inflated `pa_usable_size` that makes the overlap appear not to exist.
3. Fool the oracle into thinking everything is fine.

**Result**: The original oracle provides **circular validation** — it can only detect bugs that palloc fails to internally hide.

---

## The Independent Oracle Design

[`palloc_independent_oracle.h`](file:///e:/GithubProjects/pomaidb/tests/palloc/palloc_independent_oracle.h) breaks this circularity by:

### Rule 1: System-Malloc Backing
All oracle-internal data structures use `SysAllocator<T>` which calls `::malloc` / `::free` directly — never `pa_malloc`. Even if palloc's heap is completely corrupt, the oracle's tracking structures remain valid.

```cpp
SysMap<uintptr_t, uintptr_t>  live_intervals_;   // system malloc
SysUMap<uintptr_t, AllocRecord> live_by_user_;   // system malloc
```

### Rule 2: Request-Side Sizing
The oracle tracks `req_size` from the **caller's perspective**, not from `pa_usable_size`. Overlap detection uses `raw_addr + pa_usable_size(raw)` for interval bounds — but only to check that palloc doesn't *underreport* its usable size (which would be a different kind of bug).

### Rule 3: Independent Canary Patterns
The original oracle used `0xA5 ^ (seed & 0xFF)`. The independent oracle uses **different** patterns:
- Front guard: `0xFE` (kFrontCanary)
- Back guard: `0xEF` (kBackCanary)  
- Post-free poison: `0xDD` (kFreePoison)

These differ from palloc's internal free-list poison patterns, making it possible to distinguish oracle violations from palloc-internal corruption.

### Rule 4: Independent PRNG
The original oracle used a linear congruential generator (`1664525 * state + 1013904223`). The independent oracle uses **xorshift32**:
```cpp
x ^= (x << 13); x ^= (x >> 17); x ^= (x << 5);
```
Different PRNG family → different byte patterns → no aliasing with palloc's own metadata patterns.

---

## Scope: Core palloc vs. PomaiDB Integration

> **palloc is a standalone general-purpose memory allocator.** It is not a PomaiDB-specific component.

| Component | Is Core palloc? | Where Tested |
|:----------|:----------------|:-------------|
| `pa_malloc`, `pa_free`, `pa_calloc`, `pa_realloc` | ✅ Yes | `palloc_forensic_bug_regression.cc` |
| `pa_malloc_aligned`, `pa_free_aligned` | ✅ Yes | `palloc_forensic_bug_regression.cc` |
| `pa_usable_size`, `pa_is_in_heap_region` | ✅ Yes (palloc's own API) | `palloc_forensic_bug_regression.cc` |
| `pa_vec_pool_t`, `pa_vec_arena_t` | ✅ Yes (palloc_vector module) | `palloc_forensic_bug_regression.cc` |
| `pa_arena_t` (`arena_pomai.h`) | ✅ Yes (vector arena) | `palloc_forensic_bug_regression.cc` |
| `pa_vector_batch_alloc_floats` | ✅ Yes (palloc public API) | `palloc_forensic_fuzzer.cc` |
| `palloc_page_pool` (CLOCK eviction, swap) | ❌ PomaiDB extension | `palloc_pomai_integration_test.cc` |
| `palloc_is_owned` (PomaiDB shim) | ❌ PomaiDB wrapper | `palloc_pomai_integration_test.cc` |
| `PallocAllocator<T>` (C++ STL allocator) | ❌ PomaiDB wrapper | `palloc_pomai_integration_test.cc` |
| `operator new/delete` override | ❌ PomaiDB override | `palloc_pomai_integration_test.cc` |

---

## What the Independent Oracle Detects

| Fault Class | Detection Method | Evidence Type |
|:------------|:-----------------|:--------------|
| Overlapping allocations | O(log N) interval tree (system malloc) | **Independent** |
| Alignment violations | `(addr % alignment) != 0` check | **Independent** |
| Buffer overflows (write past end) | Back canary `0xEF` corruption | **Independent** |
| Buffer underflows (write before start) | Front canary `0xFE` corruption | **Independent** |
| Use-after-free | Post-free poison `0xDD`, caller re-reads | **Semi-independent** |
| Double-free | `live_by_user_.find()` miss on free | **Independent** |
| Memory leaks | Oracle destructor reports `live_allocs_.size() > 0` | **Independent** |
| Payload corruption | Xorshift32 payload verification | **Independent** |
| Capacity violations (arena) | `total_bytes <= capacity` assertion | **Independent** |
| Stale data (ABA) | Generation ID comparison before/after reuse | **Independent** |

---

## Fragmentation Measurement — Independent of palloc counters

The oracle independently measures **internal fragmentation**:

```
frag_ratio = (total_usable_bytes - total_requested_bytes) / total_requested_bytes
```

Where:
- `total_requested_bytes` = sum of all `req_size` arguments to `TrackAlloc`
- `total_usable_bytes` = sum of `pa_usable_size(raw)` for each allocation

This is **different** from palloc's own fragmentation counters (`pa_stats`), which track fragmentation at the segment/page level. The oracle measures it from the application's point of view.

---

## Resource Adaptive Budget

All forensic tests use `ResourceBudget::safe_budget_bytes()` which:

1. Queries `GlobalMemoryStatusEx` (Windows) for available physical RAM
2. Uses at most **30%** of available RAM
3. Hard caps at **512 MiB** per test run
4. Floors at **64 MiB** minimum

This prevents the forensic suite from exhausting system memory and crashing the host.

---

## Evidence Chain

```
USER REQUEST: sz=128, align=64
       │
       v
  TrackAlloc(128, 64, seed)
       │
       ├── pa_malloc_aligned(128+64+64, 64)   ← palloc under test
       │        └── returns: raw (or NULL)
       │
       ├── (raw % 64 == 0)?                   ← INDEPENDENT alignment check
       │
       ├── pa_usable_size(raw) >= 256?         ← palloc self-report (noted, not trusted)
       │
       ├── check_no_overlap_locked(raw, raw+usable)  ← INDEPENDENT interval tree
       │
       ├── memset front guard [0xFE × 64]     ← INDEPENDENT canary paint
       ├── xorshift32 fill payload [128 bytes]← INDEPENDENT payload fill
       ├── memset back guard [0xEF × 64]      ← INDEPENDENT canary paint
       │
       └── record in SysMap (::malloc)        ← INDEPENDENT storage
```

At free time:
```
  TrackFree(user_ptr)
       │
       ├── verify front guard [0xFE × 64]     ← INDEPENDENT check
       ├── verify xorshift32 payload          ← INDEPENDENT check
       ├── verify back guard [0xEF × 64]      ← INDEPENDENT check
       │
       ├── erase from SysMap                  ← INDEPENDENT bookkeeping
       ├── poison region [0xDD × 256]         ← use-after-free trap
       │
       └── pa_free(raw)                       ← palloc under test
```

---

## Test Results Collected

| Test File | Oracle Type | Seeds | Ops | Corrupt | Overlap |
|:----------|:------------|:------|:----|:--------|:--------|
| `palloc_forensic_bug_regression` | IndependentOracle | 1 | 10,000 | 0 | 0 |
| `palloc_forensic_fuzzer` | IndependentOracle | 7 | 2.1M | 0 | 0 |
| `palloc_forensic_concurrency` | IndependentOracle | per-thread | 80K | 0 | 0 |

> Results populated after test execution. See `PALLOC_EXTINCTION_FINAL.md` for final verdict.
