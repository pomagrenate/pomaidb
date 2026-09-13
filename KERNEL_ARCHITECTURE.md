# PomaiDB Kernel Architecture & Storage Engine Specification

## 1. Executive Summary & Kernel Mission

PomaiDB is an embedded vector database engine engineered for high-performance retrieval and low-resource environments (x86_64 AVX2/AVX-512, ARM Cortex/NEON, RISC-V).
The repository distinguishes cleanly between two fundamental subsystems:
1. **Mathematical & Vector Kernel (`pomai::core`)**: A pure, stateless computational library providing canonical SIMD-accelerated distance metrics, reference oracles, quantization encoders/decoders, and deterministic Top-K selection algorithms.
2. **Database Storage & Query Engine (`pomai::storage`, `pomai::ingest`, `pomai::query`, `pomai::routing`)**: An embedded LSM-tree-like spatial storage engine implementing durable write-ahead logging (WAL), memtables (`Rind`), immutable compacted segments (`Locule` containers `.pom`), spatial indexing (`Compass`), and multi-membrane isolation (`MembraneManager`).

---

## 2. Mathematical & Vector Kernel (`pomai::core`)

### 2.1 Canonical Distance Metrics Contract

The mathematical kernel provides a single canonical contract for all distance computations across floating-point and quantized representations:

| Metric | Function | Mathematical Definition | Range | Canonical Edge-Case Contract |
| :--- | :--- | :--- | :--- | :--- |
| **Inner Product** | `Dot(a, b)` | $\sum_{i=1}^D a_i b_i$ | $(-\infty, +\infty)$ | Returns `0.0f` if $D = 0$. |
| **Squared Euclidean** | `L2Sq(a, b)` | $\sum_{i=1}^D (a_i - b_i)^2$ | $[0, +\infty)$ | Returns `0.0f` if $D = 0$ or $a = b$. |
| **Euclidean Distance** | `L2(a, b)` | $\sqrt{\text{L2Sq}(a, b)}$ | $[0, +\infty)$ | Returns `0.0f` if $D = 0$ or $a = b$. Clamped $\ge 0$. |
| **Cosine Similarity** | `CosineSimilarity(a, b)` | $\frac{\sum a_i b_i}{\sqrt{\sum a_i^2}\sqrt{\sum b_i^2}}$ | $[-1.0, 1.0]$ | Returns `0.0f` if $\|a\| = 0$ or $\|b\| = 0$ or $D = 0$. Clamped to $[-1.0, 1.0]$. |
| **Cosine Distance** | `CosineDistance(a, b)` | $1.0 - \text{CosineSimilarity}(a, b)$ | $[0.0, 2.0]$ | Returns `1.0f` if $\|a\| = 0$ or $\|b\| = 0$. Returns `0.0f` if $D = 0$. Clamped to $[0.0, 2.0]$. |

#### Universal Query Score Invariant (`ComputeMetricScore`)
To ensure descending-sort consistency across all database stages (where higher score is always better):
- `MetricType::kInnerProduct` $\implies \text{score} = \text{Dot}(q, v)$
- `MetricType::kCosine` $\implies \text{score} = \text{CosineSimilarity}(q, v)$
- `MetricType::kL2` $\implies \text{score} = -\text{L2Sq}(q, v)$ (negated squared distance)

### 2.2 Numerical Reference Oracle (`pomai::core::ref`)
To guarantee mathematical correctness and guard against SIMD compiler regressions or precision loss, the kernel maintains an exact scalar oracle in `src/reference_distance.h` and `src/reference_distance.cc`.
- **Double-Precision Accumulation**: Uses `double` accumulators for all internal sums, products, and squared differences before casting to `float`.
- **Differential Verification**: Differential property tests (`tests/unit/kernel_differential_test.cc`) continuously verify SIMD results against `ref` across arbitrary dimensions $D \in \{0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024, 4096\}$.

### 2.3 SIMD Acceleration & Dynamic Dispatch

PomaiDB leverages [SimSIMD](https://github.com/ashvardanian/SimSIMD) for hardware-accelerated kernels alongside pure scalar fallbacks:
1. **Dynamic CPU Feature Detection**:
   - Initialized once via `pomai::core::InitDistance()`.
   - Queries hardware capabilities at runtime (`simsimd_capabilities()`) detecting AVX2, AVX-512, NEON, SVE.
2. **Punned Kernel Resolution**:
   - Uses `simsimd_find_kernel_punned(metric, datatype, capability)` to resolve function pointers.
3. **Tail Safety & Unaligned Pointers**:
   - If SimSIMD returns an exact SIMD kernel, it handles arbitrary alignments and trailing elements safely.
   - If SimSIMD cannot dispatch for a given architecture or dimension, execution transparently falls back to `pomai::core::ref` with zero degradation in numerical correctness.

### 2.4 Vector Quantization Formats

1. **Scalar Quantization (SQ8)**:
   - Compresses 32-bit floats into unsigned 8-bit integers (`uint8_t`), yielding a 4x reduction in memory footprint and bandwidth.
   - Formula: $q_i = \text{round}\left(255 \cdot \frac{x_i - \min}{\max - \min}\right)$.
   - **Numerical Hardening**:
     - Non-finite vectors ($\pm\infty$, NaN) are rejected or clamped during training.
     - Constant vectors ($x_i = c$) or empty sets with $\min \ge \max$ are guarded: $\Delta = 1.0f$, scale = $0.0f$, preventing division by zero and $+\infty$ scale overflow.
2. **1-Bit Binary Quantization**:
   - Packs coordinates into bit vectors based on sign: bit $i$ is 1 if $x_i > 0$, else 0.
   - Distance computation uses hardware POPCNT Hamming distance.
3. **Half-Precision (FP16)**:
   - IEEE 754 half-precision representation yielding 2x compression with negligible accuracy degradation.

### 2.5 Deterministic Top-K Selection Kernel (`pomai::core::topk`)

The selection kernel (`src/topk.h`) replaces naive $O(N \log N)$ sorting with bounded selection:
1. **Complexity**: $O(N + K \log K)$ average time complexity using `std::nth_element` partition followed by sorting only the top $K$ items.
2. **Deterministic Tie-Breaking (`TopKScoreDescIdAsc`)**:
   - When scores differ: higher score ranks first (`score_a > score_b`).
   - When scores are identical (e.g. equidistant vectors or duplicate scores): tie is deterministically broken by vector ID ascending (`id_a < id_b`).
   - Guarantees strict reproducibility across query executions.
3. **Bounded Heap Selection (`BoundedTopKQueue`)**:
   - Min-heap of size $K$ for streaming ingestion or multi-way merge, discarding candidates worse than the current $K$-th best.

---

## 3. Database Engine Subsystems & Architecture

### 3.1 Subsystem Boundaries

```
                 ┌──────────────────────────────────────┐
                 │          C API & Bindings            │
                 │      (Python, JS, Rust, Go)          │
                 └──────────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │          MembraneManager             │
                 │   (Namespaces, Concurrency Mutex)    │
                 └─────────┬──────────────────┬─────────┘
                           │                  │
               ┌───────────▼────────┐   ┌─────▼──────────────┐
               │    Write Path      │   │     Read Path      │
               │  (Ingest / Compaction) │ (Query Execution)  │
               └───────────┬────────┘   └─────┬──────────────┘
                           │                  │
        ┌──────────────────┼──────────────────┼──────────────────┐
        │                  │                  │                  │
 ┌──────▼──────┐    ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐
 │ Write-Ahead │    │ Active Mem-  │   │ Spatial      │   │ Compacted    │
 │ Log (WAL)   │    │ Table (Rind) │   │ Index        │   │ Segments     │
 │             │    │              │   │ (Compass)    │   │ (Locules)    │
 └─────────────┘    └──────┬───────┘   └──────────────┘   └──────┬───────┘
                           │                                     │
                    ┌──────▼─────────────────────────────────────▼──────┐
                    │               Compaction Press                    │
                    │      (Flushes Frozen Rinds to Locule .pom)        │
                    └───────────────────────────────────────────────────┘
```

### 3.2 Component Responsibility Matrix

| Subsystem | Components | Primary Responsibility |
| :--- | :--- | :--- |
| **Foreign Interface** | `capi/` (`capi_db.cc`, `c_api.h`) | Exposes thread-safe C ABI; manages handle validation and prevents UAF. |
| **Membrane Management** | `MembraneManager` | Multi-tenant namespace router; manages lifetime of independent databases with `std::shared_mutex`. |
| **Write-Ahead Log** | `storage::Wal` | Append-only transaction log with per-record CRC32C and fsync control for crash recovery. |
| **Active MemTable** | `ingest::Rind` | Concurrent in-memory buffer (`std::mutex`); enforces finite vector validation and memory budgeting. |
| **Compaction Engine** | `compaction::Press` | Background/explicit merger converting frozen memtables into immutable `.pom` Locules; purges tombstones. |
| **Segment Container** | `storage::Locule` | Memory-mapped on-disk container; verifies header and Aril directory CRC32C checksums. |
| **Segment Block Storage** | `storage::ArilReader`, `ArilWriter` | Columnar vector data within `.pom`; houses Pulp, Seed Kernel, Seed Scar, and Seed Directory. |
| **Spatial Manifest** | `manifest::FruitMap` | Atomic generational catalog (`FruitSnapshot`) providing lock-free snapshot isolation for readers. |
| **Spatial Routing** | `routing::Compass` | Hierarchical clustering; maintains centroid vectors and bounding spheres to prune distant Locules. |
| **Query Engine** | `query::PomegranateQuery` | Orchestrates 5-stage search (`Orient -> Peel -> Taste -> Pick -> Rerank`). |
| **Vector Index** | `index::HnswIndex` | Standalone hierarchical navigable small world graph index for high-recall ANN queries. |

---

## 4. Storage Engine Specification: The Pomegranate Model

### 4.1 Locule File Layout (`.pom`)

On-disk segments are packaged into immutable **Locule** files:

```
+------------------------------------------------------------------------+
|                      Locule Container File (*.pom)                     |
+------------------------------------------------------------------------+
| PomaiFileHeader (64 bytes)                                             |
|   - Magic: 0x504F4D4149444200 ("POMAIDB\0")                            |
|   - Format Version: 1                                                  |
|   - Generation ID (uint64)                                             |
|   - Dimension (uint32)                                                 |
|   - Metric Type (uint8)                                                |
|   - Quantization Mode (uint8)                                          |
|   - Aril Count (uint32)                                                |
|   - Directory Offset (uint64)                                          |
|   - Directory Size (uint64)                                            |
|   - Directory CRC32C (uint32)                                          |
+------------------------------------------------------------------------+
| Aril 1 Block Content (Aligned to 64 bytes)                             |
|   - Aril Block Header (Magic, Slot Count, Tombstone Count)             |
|   - Pulp View: Quantized codes (SQ8 / FP16)                            |
|   - Seed Kernel View: Contiguous exact FP32 vectors                    |
|   - Seed Scar View: Tombstone bitset for deleted vectors               |
|   - Seed Directory: Inverted vector_id -> slot index table             |
|   - Metadata Payload: Variable-length user attributes and timestamps   |
+------------------------------------------------------------------------+
| Aril 2 ... Aril N Block Content                                        |
+------------------------------------------------------------------------+
| Aril Directory (Array of ArilDirectoryEntry)                           |
|   - Aril ID, Offset, Length, Block CRC32C                              |
+------------------------------------------------------------------------+
| Locule Footer (64 bytes)                                               |
|   - Centroid Vector (float[dim])                                       |
|   - Bounding Radius (float)                                            |
|   - Magic Footer: 0x454E445F504F4D41 ("END_POMA")                      |
+------------------------------------------------------------------------+
```

### 4.2 Data Integrity: Dual-Layer CRC32C

1. **Directory Protection**: `PomaiFileHeader` stores `directory_crc32`. If the directory table is corrupted, `Locule::Open` fails immediately.
2. **Block Protection**: Each `ArilDirectoryEntry` stores the CRC32C of its corresponding Aril block. Data corruption inside a block is trapped before SIMD memory access.

---

## 5. Pipelined Query Execution (5 Stages)

Queries in PomaiDB follow a 5-stage pipeline designed to minimize disk I/O and FP32 floating-point calculations:

```
[Query Vector q]
       │
       ▼
 Stage 1: Orient  ──► Compass scores centroid distances; ranks Locules by proximity
       │
       ▼
 Stage 2: Peel    ──► Selects candidate Locules; skips locules outside bounding sphere
       │
       ▼
 Stage 3: Taste   ──► Scans quantized Pulp (SQ8) in active Locules + in-memory Rinds;
       │              produces Top-M coarse candidates (M = K * candidate_expansion)
       │
       ▼
 Stage 4: Pick    ──► Deduplicates candidates across Rind and Locule snapshots;
       │              resolves newest-wins MVCC semantics; filters tombstones (Seed Scar)
       │
       ▼
 Stage 5: Rerank  ──► Reads exact FP32 Seed Kernel vectors for the top-M candidates;
       │              computes exact metric scores; executes deterministic SelectTopK
       ▼
[Final Top-K Results]
```

---

## 6. Concurrency, Memory Ownership & Invariants

### 6.1 Concurrency Model

1. **Membrane Catalog**:
   - Protected by `std::shared_mutex membranes_mu_`.
   - Creation/deletion of membranes requires exclusive lock (`std::unique_lock`).
   - Querying or inserting vectors acquires shared access (`std::shared_lock`), allowing full parallelism across membranes.
2. **MemTable (`Rind`)**:
   - Synchronized internally via `std::mutex mu_`.
   - Append operations take brief locks to record to WAL and insert into memtable structures.
3. **Generational Snapshots (`FruitMap`)**:
   - Manifest publishes immutable `FruitSnapshot` references via atomic pointer swap (`std::atomic<std::shared_ptr<FruitSnapshot>>`).
   - Readers acquire snapshot pointers with zero mutex contention; compaction threads update the manifest asynchronously.

### 6.2 Durability & Crash Recovery Invariants

1. **Write-Ahead Logging**:
   - Every mutation is appended to the active WAL file with a CRC32C checksum prior to memtable insertion.
   - On startup, WAL replay reconstructs uncompacted memtables. Corrupt tail records (partial writes from power failures) are safely discarded.
2. **Atomic Segment Publishing**:
   - Compaction writes new `.pom` segments to temporary files (`*.tmp`).
   - Segments are made permanent via atomic filesystem rename (`env->RenameFile`), preventing partial segment visibility.

---

## 7. Approximate Nearest Neighbor (ANN) Production Reality

### 7.1 Architecture Findings & Forensics

A critical finding from the kernel architectural audit is the role of HNSW versus Compass in production queries:
1. **Production Path**:
   - The primary database search path (`PomegranateEngine::Search` $\to$ `PomegranateQuery::Execute`) routes through **Compass spatial clustering + Locule Pulp SQ8 scan + Seed Kernel FP32 reranking**.
   - This architecture is optimized for embedded devices: zero graph construction latency on ingest, instant cold start via memory-mapped `.pom` segments, and low RAM consumption.
2. **HNSW Index (`HnswIndex`)**:
   - `pomai::index::HnswIndex` is a fully functional, self-contained in-memory HNSW implementation (featuring $M$, $ef_{construction}$, $ef_{search}$).
   - It provides high Recall@K on static workloads but is decoupled from the main dynamic LSM-tree query path.

### 7.2 Empirical Crossover Points

Empirical evaluation across dataset sizes $N \in \{1\text{K}, 5\text{K}, 10\text{K}, 25\text{K}\}$ ($D=64$, $K=100$):
- **$N \le 5\text{K}$**: Flat SIMD brute force delivers sub-millisecond p50 latency with 100% recall. The overhead of graph traversal or centroid scoring exceeds raw SIMD scanning throughput.
- **$5\text{K} < N \le 25\text{K}$**: Compass routing + Pulp flat scan achieves 3x-8x higher QPS than unindexed flat scan while retaining $>0.92$ Recall@10.
- **$N > 25\text{K}$**: HNSW graph traversal provides logarithmic query scaling ($O(\log N)$), outperforming flat partitioning on large static datasets at the expense of memory footprint and construction time.
