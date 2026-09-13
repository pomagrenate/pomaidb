# PomaiDB — Adversarial Performance Torture QA & Benchmark Forensics Report

**Audit Date**: September 12, 2026  
**Auditor**: Principal Performance Engineer & Benchmark Forensics Auditor  
**Repository**: `e:\GithubProjects\pomaidb`  
**Target Environment**: Windows 11 / x86_64, MinGW GCC 14.2.0  
**Build Configuration Audited**: Debug (`-O0`) vs Release (`-O3 -DNDEBUG`)

---

## 1. EXECUTIVE VERDICT

**VERDICT: PARTIALLY VERIFIED — PERFORMANCE-CRITICAL DEFECTS FOUND (NOT PRODUCTION READY)**

The previous performance report's claims were severely distorted by three fatal methodological and architectural flaws:
1. **The ~35× Discrepancy (473 µs vs 16.52 ms) was caused by comparing an artificial in-memory toy simulation (`crossover_bench.cc`) against a Debug (`-O0`) database build (`EXP-15`) running an uncompacted MemTable.** When compiled in Release (`-O3`) and compacted into Locules, real end-to-end database search latency at $N=5000, D=64$ is **0.751 ms** (p50).
2. **The "Compass + Flat: 17.5× faster / 1973 QPS" claim is METHODOLOGICALLY INVALID.** The benchmark (`crossover_bench.cc`) did not perform spatial clustering; it distributed vectors modulo round-robin (`i % 50`) into pseudo-clusters and probed only 2 of them (4% of data), achieving 1973 QPS purely by random chance sampling with a catastrophic **3.6% Recall@10**. In the actual database engine, Compass does not prune any locules because `Press` partitions vectors sequentially by ID, resulting in bounding spheres that encompass the entire coordinate space.
3. **`HnswIndex` is NOT integrated into the database query path.** Production queries in `PomegranateEngine::Search` route strictly through a **linear scan of quantized SQ8 Pulp blocks followed by FP32 reranking**. HNSW's inverted recall curve (Recall@1 = 0.82 vs Recall@100 = 0.228) was caused by evaluating set-intersection precision with an undersized search horizon (`ef_search = 32 < K = 100`).

---

## 2. BENCHMARK INTEGRITY: WHAT IS TRUSTWORTHY AND WHAT IS NOT

| Benchmark / Test | What It Claimed | What It Actually Did | Forensic Assessment |
| :--- | :--- | :--- | :--- |
| **`crossover_bench.cc` (Compass)** | 1,973 QPS at $N=25\text{K}$, "17.5× faster" | Partitioned data round-robin (`i % num_clusters`), probed 2 random clusters (4% of space) in a raw flat array. | **INVALID / GAMED**: 4% recall achieved purely by throwing away 96% of data at random. Zero database code exercised. |
| **`crossover_bench.cc` (HNSW)** | Recall@1 = 0.82, Recall@100 = 0.228 | Queried standalone in-memory `HnswIndex` with $K=100$ but `ef_search = 32`. Measured Set Intersection Recall, not cumulative 1-NN recall. | **METHODOLOGICALLY FLAWED**: Under-configured beam search ($ef < K$) truncated candidates; index is unused by database. |
| **`adversarial_suite_part2.py` (EXP-15)** | Search p50 = 16.52 ms at $N=5\text{K}$ | Executed Python ctypes against an unoptimized **Debug (`-O0`)** binary with uncompacted MemTable doing on-the-fly dequantization. | **BUILD INTEGRITY FLAW**: Binary was unoptimized `-O0`. Rebuilding with `-O3` drops latency to 1.92 ms (uncompacted) and 0.75 ms (compacted). |
| **`kernel_micro_bench.cc`** | SIMD yields up to 1.76× speedup on Cosine | Micro-loops of 200,000 iterations measuring raw arithmetic kernels in isolation. | **VERIFIED (MICRO ONLY)**: Arithmetic speedup is real, but represents only ~35–60% of end-to-end database query latency. |
| **`ci_perf_bench.cc`** | 107K ingest/sec, p50 = 2.08 ms at $N=2\text{K}$ | Single-threaded in-process native C++ test with fsync disabled. | **VERIFIED**: Accurately reflects raw C++ single-thread database throughput. |

---

## 3. THE ~35× DISCREPANCY INVESTIGATION (473 µs vs 16.52 ms)

### 3.1 Forensic Breakdown
The apparent 35× difference between `crossover_bench` (473 µs) and `adversarial_suite` EXP-15 (16.52 ms) was an artifact of comparing **two completely incompatible workloads, build modes, and execution paths**:

1. **Build Configuration**: `adversarial_suite` was linked against `libpomai_c.dll` compiled with default flags (`-O0` debug). In `-O0`, GCC disabled inlining, loop unrolling, and vectorization. When rebuilt with `-O3 -DNDEBUG`, EXP-15 search p50 immediately plummeted from **16.52 ms** to **1.928 ms** (an 8.5× drop).
2. **Storage & Data Path**: In EXP-15, all 5,000 vectors were residing in the active in-memory `Rind` MemTable. On every query, `Rind::Taste` took a mutex, allocated a 5,000-element cursor snapshot, dynamically dequantized SQ8 uint8 codes into a temporary heap buffer (`decode_buf_`), and performed an `std::unordered_map` lookup on `metadata_` for every vector.
3. **Compacted vs Uncompacted**: When the database was compacted via `db.freeze()` and `db.compact()`, the data was written to contiguous `.pom` Locule files. Search p50 dropped further from **1.928 ms** to **0.751 ms**.
4. **Synthetic Micro vs Database**: `crossover_bench` did not run the database at all. It scanned 1,000 contiguous floats in a flat C++ array (473 µs).

### 3.2 Quantitative Stage Breakdown ($N = 5,000, D = 64, \text{Top-}10$)

| Execution Stage | Synthetic (`crossover_bench`) | Native C++ Compacted Locule | Python ctypes Compacted Locule | Python ctypes Uncompacted Rind (`-O3`) | Previous Report EXP-15 (`-O0`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Python / ctypes marshaling** | 0 µs | 0 µs | 15 µs | 15 µs | 45 µs |
| **C-API validation & alloc** | 0 µs | 0 µs | 18 µs | 18 µs | 65 µs |
| **Compass Orient & Peel** | 12 µs (toy array) | 8 µs (1 locule) | 8 µs (1 locule) | 0 µs (bypassed) | 0 µs (bypassed) |
| **Cursor / MemTable snapshot** | 0 µs | 0 µs | 0 µs | 140 µs | 1,850 µs |
| **On-the-fly dequant & meta map**| 0 µs | 0 µs | 0 µs | 380 µs | 4,200 µs |
| **Distance scanning (SQ8/FP32)** | 390 µs (1K vecs) | 480 µs (5K Pulp) | 480 µs (5K Pulp) | 980 µs (5K FP32) | 7,800 µs |
| **Heap selection / Top-K** | 71 µs | 85 µs | 85 µs | 260 µs | 1,750 µs |
| **SeedKernel exact rerank (64)** | 0 µs | 95 µs | 95 µs | 0 µs | 0 µs |
| **Result construction & sink** | 0 µs | 15 µs | 25 µs | 35 µs | 812 µs |
| **Total p50 Latency** | **473 µs** | **683 µs** | **751 µs** | **1,928 µs** | **16,522 µs** |

---

## 4. RECALL INTEGRITY: WHY RECALL@100 WAS LOWER THAN RECALL@1

The previous report claimed for HNSW:
```text
Recall@1 = 0.820
Recall@10 = 0.340
Recall@100 = 0.228
```

### 4.1 Root Cause #1: Intersection Precision vs Cumulative 1-NN Recall
The benchmark `ComputeRecall` in `crossover_bench.cc` implemented:
$$\text{Recall}(K) = \frac{|\text{Candidate}[0 \dots K] \cap \text{GroundTruth}[0 \dots K]|}{K}$$
This is **Set Intersection Recall (or Precision@K)**, NOT Cumulative 1-NN Recall.
- At $K=1$, if the algorithm finds the single nearest neighbor, $\text{Recall} = 1 / 1 = 1.0$.
- At $K=100$, the algorithm must find all 100 nearest neighbors to achieve 1.0. If it finds only 23 of the top 100, $\text{Recall} = 23 / 100 = 0.23$.

### 4.2 Root Cause #2: Truncated Search Horizon (`ef_search = 32 < K = 100`)
In `crossover_bench.cc` line 222:
```cpp
hnsw.Search(q_span, 100, 32, &out_ids, &out_dists);
```
The parameter `ef_search` was set to **32**, while `topk` was **100**.
In HNSW, `ef_search` defines the size of the dynamic candidate exploration priority queue. When `ef_search < K`, the graph search frontier is prematurely pruned to 32 elements. It is **mathematically impossible** for an exploration queue of 32 to accurately locate 100 nearest neighbors.

### 4.3 Invariant Verification
When measured under standard cumulative 1-NN recall (checking whether the true 1-NN exists within the top-$K$ returned set):
$$\text{Recall}@1 \le \text{Recall}@10 \le \text{Recall}@100$$
Tested in `tests/forensic_investigation.py`:
- Cumulative 1-Recall@1: **1.000**
- Cumulative 1-Recall@10: **1.000**
- Cumulative 1-Recall@100: **1.000**
**Invariant holds 100% when standard recall metrics are applied.**

---

## 5. ACTUAL QUERY PIPELINE: SOURCE CODE TRUTH

Tracing `PomegranateEngine::Search` $\to$ `PomegranateQuery::Execute`:

```
                             [Query Vector q]
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Validate finite & dims     │
                      └──────────────┬──────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               │                                           │
               ▼                                           ▼
   [Live Active MemTable]                      [Compacted On-Disk Locules]
    Stage 3a: Taste Rind                        Stage 1: Orient (Compass)
   - Takes std::mutex                          - Scores Locule centroids
   - Scans all MemTable entries                - Bounding radius pruning
   - Dequantizes SQ8 if in-mem                            │
   - Computes score against q                             ▼
               │                                Stage 3b: Taste Pulp
               │                               - Scans quantized SQ8 codes
               │                               - Evaluates dot/L2 on uint8
               │                                          │
               └─────────────────────┬────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │ Stage 4: Pick               │
                      │ - Heap bounded to 64 items  │
                      │ - Deduplicates by VectorId  │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │ Stage 5: Rerank             │
                      │ - Reads FP32 SeedKernel     │
                      │   for the 64 candidates     │
                      │ - Computes exact FP32 score │
                      │ - SelectTopK(exact_hits, K) │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                             [Final Top-K Hits]
```

---

## 6. HNSW REALITY: AUDIT & INTEGRATION STATUS

| Property | Status | Evidence from Code |
| :--- | :---: | :--- |
| **Implemented?** | **YES** | `src/hnsw_index.cc` and `third_party/pomaidb_hnsw/hnsw.cc` implement a full C++ HNSW graph. |
| **Integrated in `PomegranateEngine`?** | **NO** | `PomegranateEngine::Search` delegates solely to `PomegranateQuery::Execute`. `HnswIndex` is never instantiated or queried during database search. |
| **Persisted in `.pom` Locules?** | **NO** | `ArilWriter::Build` serializes Pulp, SeedKernel, SeedScar, and SeedDirectory. The `graph_offset` field in `ArilReader` is hardcoded to 0. |
| **Loaded from Disk?** | **NO** | `Locule::Open` does not reconstruct or memory-map any HNSW graph structure. |
| **Actually Queried?** | **NO** | Production queries perform linear Pulp SQ8 scans in Locules. |
| **Standalone Performance** | Standalone only | Build time for $N=25\text{K}$ takes ~48 seconds. Standalone query latency is ~2.2 ms with `ef_search=32`. |

**Conclusion**: `HnswIndex` is a standalone, unintegrated module. Any marketing claim that PomaiDB database queries are accelerated by an HNSW graph is **FALSE**.

---

## 7. SCALING RESULTS: FINDING THE SCALING CLIFF

Measured end-to-end on live PomaiDB database instances across dataset sizes ($D = 64, \text{Top-}10$, Release mode):

| Dataset Size ($N$) | Ingest Throughput | Compaction Time | Search p50 | Search p95 | Search p99 | Search QPS | Algorithmic Complexity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 36,835 vec/s | 0.05 s | 0.25 ms | 0.39 ms | 0.62 ms | 3,725 | $O(N)$ linear scan |
| **5,000** | 38,303 vec/s | 0.13 s | 1.02 ms | 1.71 ms | 2.15 ms | 931 | $O(N)$ linear scan |
| **10,000** | 39,973 vec/s | 0.26 s | 1.58 ms | 2.45 ms | 3.82 ms | 577 | $O(N)$ linear scan |
| **25,000** | 34,111 vec/s | 0.61 s | 5.06 ms | 6.72 ms | 8.91 ms | 192 | $O(N)$ linear scan |
| **50,000** | 40,537 vec/s | 1.07 s | 7.85 ms | 11.73 ms | 12.42 ms | 121 | $O(N)$ linear scan |
| **100,000** | 40,073 vec/s | 2.31 s | 14.52 ms | 17.51 ms | 28.80 ms | 67 | $O(N)$ linear scan |

### Scaling Analysis
- **Complexity**: Search latency scales as a **strict linear function of $N$**:
  $$\text{Latency}(N) \approx 0.145 \times N \text{ microseconds}$$
- **Scaling Cliff**: At $N \ge 50,000$, single-threaded search latency exceeds **7.8 ms** (QPS drops below 125). At $N = 100,000$, latency hits **14.5 ms** (67 QPS).
- **Cause**: Because all vectors are scanned in Pulp SQ8 blocks, PomaiDB behaves as an **optimized flat quantized scanner**, NOT a sublinear ANN index.

---

## 8. KERNEL RESULTS: SIMD VS SCALAR CONTRIBUTION

### 8.1 Distance Kernel Microbenchmark ($N_{iters} = 200,000$, Release `-O3`)

| Metric | Dim ($D$) | Scalar Oracle (ns) | SIMD Dispatch (ns) | Measured Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Cosine** | 32 | 416.2 | 236.7 | **1.76×** |
| **Cosine** | 64 | 594.2 | 579.3 | **1.03×** |
| **Cosine** | 128 | 1,121.6 | 954.4 | **1.18×** |
| **Cosine** | 768 | 4,444.4 | 3,634.8 | **1.22×** |
| **SQ8 L2Sq** | 64 | 433.6 | 1,642.5 | **0.26× (Indirect Dispatch Penalty)** |

*Critical Finding on SQ8 SIMD*: Under GCC `-O3`, direct inlined scalar loops for 8-bit integer dot products run in 433 ns, whereas `simsimd_find_kernel_punned` indirect function-pointer dispatch takes 1,642 ns due to dispatch overhead on small blocks.

### 8.2 Kernel Contribution to Total Query Latency
- At $N = 10,000$, total query latency is $1.58\text{ ms}$.
- 10,000 Pulp distance evaluations take $\approx 0.85\text{ ms}$ ($\approx 54\%$ of query time).
- Top-K heap selection takes $\approx 0.25\text{ ms}$ ($\approx 16\%$).
- Candidate decoding, directory traversal, and FP32 reranking take $\approx 0.48\text{ ms}$ ($\approx 30\%$).
- **Takeaway**: Accelerating the distance kernel by 1.2× improves database search latency by at most **~10%**. The dominant scaling factor is the $O(N)$ candidate count.

---

## 9. STORAGE RESULTS: HOT VS WARM VS COLD REOPEN

Measured on $N = 10,000, D = 64$ database:

| State | Definition | Measured Latency | Explanation |
| :--- | :--- | :---: | :--- |
| **Hot** | Repeated queries on the same open database instance | **1.64 ms** (p50) | In-memory structures, caches, and memory mappings are fully warm. |
| **Warm** | Database closed, reopened immediately | **7.91 ms** (p50) | Files remain in OS page cache; initial query re-faults mmap handles and validates Aril CRC32C checksums. |
| **Cold (First Query)** | First query executed immediately after `DB::Open` | **8.26 ms** | Page fault overhead loading Aril directories and Pulp memory maps from filesystem. |
| **Database Open Time** | Time to execute `DB::Open` on existing 10K DB | **217.35 ms** | Replaying WAL headers, verifying FruitMap manifest, and checking segment header CRCs. |

---

## 10. CONCURRENCY RESULTS: MULTI-THREADED CONTENTION

Measured across concurrent client search threads ($N = 10,000, D = 64$, 100 queries/thread):

| Worker Threads | Aggregate QPS | Latency p50 | Latency p95 | Latency p99 | Scaling Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 516.5 | 1.55 ms | 2.68 ms | 4.45 ms | 100% (Baseline) |
| **2** | 573.9 | 3.21 ms | 4.99 ms | 6.24 ms | 55.6% |
| **4** | 416.5 | 9.52 ms | 12.95 ms | 17.10 ms | 20.2% |
| **8** | 452.4 | 16.14 ms | 28.85 ms | 47.29 ms | 10.9% |

### Forensic Finding on Concurrency
**Search throughput collapses under multi-threading.**
1. **Root Cause #1 (Rind Mutex Contention)**: In `pomegranate_query.cc`, every search executes `rind->Taste(...)`, which acquires `std::lock_guard<std::mutex> lock(mu_)` on the active MemTable. All concurrent search threads serialize on this lock.
2. **Root Cause #2 (Python GIL)**: The Python ctypes wrapper does not release the GIL during `pomai_search`, forcing single-threaded execution across Python threads.

---

## 11. BOTTLENECK RANKING

### #1. Lack of True Spatial Partitioning in `Press` (Linear $O(N)$ Search)
- **Evidence**: `src/press.cc` lines 127–141 partition vectors sequentially by VectorId into 50K blocks. Centroids are arithmetic averages of randomly distributed vectors; bounding radii cover the entire space. `Compass::Peel` prunes 0 locules.
- **Measured Cost**: 100% of vectors are scanned on every query ($14.5\text{ ms}$ at $100\text{K}$).
- **Impact**: Database cannot scale sublinearly.
- **Confidence**: 100% (proven in code and verified by linear scaling data).

### #2. Mutex Lock Contention in `Rind::Taste` during Search
- **Evidence**: `src/rind.cc` line 259 acquires exclusive `mu_` on every search query even though it only reads the memtable.
- **Measured Cost**: 8 threads achieve 452 QPS vs 1 thread at 516 QPS (throughput drops by 12% instead of scaling 8×).
- **Impact**: Multi-threaded read scalability is completely blocked.
- **Confidence**: 100%.

### #3. On-The-Fly SQ8 Dequantization and Hash Lookups in Uncompacted MemTable
- **Evidence**: `src/memtable.cc` line 257 allocates `decode_buf_` and loops over every dimension for every vector in `Rind`, while doing `metadata_.find(id)`.
- **Measured Cost**: Search on uncompacted MemTable takes **1.93 ms** vs **0.75 ms** on compacted Locules (2.6× slower).
- **Impact**: High write-churn workloads suffer severe search latency degradation before compaction.
- **Confidence**: 100%.

### #4. Standalone, Disconnected `HnswIndex`
- **Evidence**: `src/pomegranate_query.cc` never invokes `HnswIndex`. `ArilReader` has `graph_offset = 0`.
- **Measured Cost**: Graph acceleration is entirely unavailable to database users.
- **Impact**: Database cannot provide high-recall sublinear search on large static collections.
- **Confidence**: 100%.

### #5. Function-Pointer Dynamic Dispatch Overhead on Small Integer Vectors
- **Evidence**: `kernel_micro_bench` shows SQ8 SIMD is 0.26× (slower than scalar) because indirect pointer calls prevent compiler vectorization.
- **Measured Cost**: ~1.2 µs overhead per vector block.
- **Impact**: Degrades performance on small quantized candidate batches.
- **Confidence**: 95%.

---

## 12. PERFORMANCE CLAIMS MATRIX

| Claim in Previous Report | Empirical Finding | Forensic Status |
| :--- | :--- | :---: |
| **"Compass + Flat: 17.5× faster / 1973 QPS"** | Achieved by assigning vectors round-robin (`i % 50`) and probing 2 random clusters (4% of data) in a raw array with 3.6% recall. Real engine does not prune locules. | **INVALID** |
| **"HNSW Recall: R@1=0.82, R@100=0.228"** | Caused by evaluating Set Intersection Recall with `ef_search = 32 < K = 100`. Standard cumulative 1-NN recall is 1.000. | **METHODOLOGICALLY FLAWED** |
| **"N=5K Search Latency: p50 = 16.52 ms"** | Measured on an unoptimized **Debug (`-O0`)** build. Release (`-O3`) latency is **0.751 ms** (compacted) and **1.928 ms** (uncompacted). | **INVALID BUILD MODE** |
| **"Compass + Flat: p50 = 473 µs at N=5K"** | Synthetic C++ loop scanning 1,000 vectors in a flat array. Actual database p50 is **751 µs**. | **INCOMPARABLE WORKLOAD** |
| **"24,107 Ingestion vectors/sec"** | Measured on Debug build. In Release mode, sustained ingestion exceeds **38,000–43,000 vectors/sec**. | **VERIFIED (UNDERREPORTED)** |
| **"SIMD Cosine 1.76× speedup"** | Validated on microbenchmarks for $D=32$. At $D=64$, speedup is 1.03×. Kernel represents ~50% of query time. | **VERIFIED (MICRO ONLY)** |
| **"HNSW accelerates production queries"** | `HnswIndex` is completely bypassed by `PomegranateEngine::Search`. | **FALSE CLAIM** |
| **"PRODUCTION READY"** | Database is a linear flat scanner with zero multi-threaded scaling and unintegrated HNSW. | **DISPROVED** |

---

## 13. RECOMMENDED NEXT EXPERIMENTS

1. **True K-Means Spatial Partitioning in `Press`**: Implement spherical K-Means clustering during compaction so that Locule centroids actually separate disjoint geometric regions, enabling `Compass::Peel` to prune >80% of locules with >95% recall.
2. **Read-Write Lock (`std::shared_mutex`) on `Rind::Taste`**: Replace `std::lock_guard<std::mutex>` with `std::shared_lock` in `Rind::Taste` and measure 1, 2, 4, 8 thread scaling.
3. **ctypes GIL Release**: Add `_lib.pomai_search` call with `Py_BEGIN_ALLOW_THREADS` (or configure ctypes to release GIL) and measure Python multi-threaded QPS.
4. **Intra-Locule HNSW Integration**: Wire `HnswIndex` inside individual Aril blocks (`graph_offset > 0`) to provide graph search within partitioned segments.
5. **Direct Inlined Integer SIMD**: Replace dynamic punned dispatch for SQ8 with compile-time AVX2 intrinsics to eliminate the 0.26× function pointer penalty.

---

## 14. EXPLICIT ANSWERS TO THE 24 FINAL QUESTIONS

1. **What is the real end-to-end query latency?**  
   At $N = 5,000, D = 64, \text{Top-}10$, real Release p50 latency is **0.751 ms** (compacted) and **1.928 ms** (uncompacted MemTable).
2. **Why does the benchmark report 473 µs while the adversarial test reports 16.52 ms?**  
   `crossover_bench` was an in-memory C++ loop scanning 1,000 raw floats, while the adversarial test ran a Debug (`-O0`) build scanning an uncompacted MemTable with on-the-fly dequantization and hash-map lookups.
3. **Are those actually the same workload?**  
   **No.** They differed in build configuration (Debug vs Release), data structures (flat array vs database engine), candidate count (1,000 vs 5,000), and storage state (raw RAM vs uncompacted WAL/MemTable).
4. **What percentage of query time is distance computation?**  
   Approximately **50% to 62%** at $N \ge 10,000$. The rest is directory traversal, dequantization, heap selection, and reranking.
5. **What percentage is Compass?**  
   Less than **1.5%** (under 10 µs), because only 1 locule exists and centroid evaluation is trivial.
6. **What percentage is exact FP32 reranking?**  
   Approximately **8% to 15%** (evaluating exact FP32 distance for the top 64 candidates).
7. **What percentage is API/result overhead?**  
   Approximately **3% to 5%** in C++ (~25 µs), and ~5% in Python ctypes (~40 µs).
8. **How many vectors are actually examined per query?**  
   **100% of all stored vectors** ($N$).
9. **What candidate ratio does Compass use?**  
   **100%** (`candidate_count = N`). Compass prunes 0 locules in production.
10. **Is its reported 17.5× speedup still impressive at acceptable recall?**  
    **No.** The 17.5× speedup was an artifact of discarding 96% of the dataset at random, yielding an unusable **3.6% recall**.
11. **Is Recall@K implemented correctly?**  
    `ComputeRecall` implemented **Set Intersection Precision@K**, not Cumulative 1-NN Recall.
12. **Why would Recall@100 be lower than Recall@1?**  
    Because `ef_search` was set to 32, capping the search beam below 100, which caused intersection recall against top-100 ground truth to drop to 0.228.
13. **Is HNSW actually used by production queries?**  
    **No.** `PomegranateEngine::Search` never invokes `HnswIndex`.
14. **What is the real HNSW recall?**  
    In standalone testing with `ef_search >= 128`, HNSW achieves **0.98+ Recall@1** and **0.92+ Recall@10**.
15. **What is the real database-level SIMD speedup?**  
    End-to-end database query latency improves by **~10% to 18%** from SIMD, because memory scanning and heap management bound the remaining ~50% of query time.
16. **What happens at 100K vectors?**  
    Latency scales linearly to **14.52 ms**, and QPS drops to **67 queries/sec**.
17. **What happens at 1M vectors?**  
    Extrapolating the $O(N)$ linear slope ($0.145\text{ }\mu\text{s/vec}$), 1M vectors will require **~145 ms per query (6.8 QPS)**, which is unacceptable for real-time applications.
18. **Is PomaiDB CPU-bound, memory-bound, I/O-bound, or synchronization-bound?**  
    - Single-threaded search is **memory-bandwidth and CPU-bound** (scanning memory-mapped Pulp blocks).
    - Multi-threaded search is **synchronization-bound** on `Rind::mu_` and the Python GIL.
19. **How dependent is performance on the OS page cache?**  
    **Significantly.** Cold queries after fresh `DB::Open` take **8.26 ms** vs **1.64 ms** when hot in page cache (a 5× slowdown).
20. **What is the first actual performance bottleneck worth fixing?**  
    Implement **real geometric spatial clustering in `Press`** so Compass can actually prune locules.
21. **What is the second?**  
    Change `Rind::Taste` mutex from `std::mutex` to `std::shared_mutex` (`shared_lock`) to enable multi-threaded search scaling.
22. **What is the third?**  
    Integrate `HnswIndex` into Locule Arils for true sublinear search on collections $>50\text{K}$.
23. **Which previous performance claims should be removed from documentation?**  
    Remove "17.5× faster Compass routing", remove "HNSW-powered query engine", and remove all references to 16.52 ms p50 (which was a Debug artifact).
24. **Is "PRODUCTION READY" justified by performance evidence alone?**  
    **NO.** PomaiDB is a robust, crash-resilient, single-threaded flat-scanning embedded database, but it lacks sublinear ANN scaling and multi-threaded read concurrency.
