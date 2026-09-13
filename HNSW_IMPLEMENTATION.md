# PomaiDB — Production HNSW Implementation Architecture

## 1. Executive Summary

This document describes the production implementation of Hierarchical Navigable Small World (HNSW) indexing and spatial Locule execution in PomaiDB. 

Following the findings of Rounds 1–3 performance forensics, PomaiDB has transitioned from an $O(N)$ sequential flat-scan architecture to a production sub-linear vector database engine combining:
1. **Zero-Lock Rind Concurrency**: Point-in-time tombstone snapshots eliminating read-path mutex serialization.
2. **Balanced Spatial Partitioning**: Geometric K-Means clustering in Press establishing tight bounding spheres around Locules.
3. **Upstream Algorithmic Authority**: Direct vendoring of `nmslib/hnswlib` (commit `058d7a866e462c00f0a6dea8660969379d5916bd`) with zero custom layer heuristics or altered edge selection.
4. **Zero-Copy Persistence**: Embedding serialized HNSW graph blocks inside immutable `.pom` Aril files with CRC32C validation and graceful fallback to SIMD Pulp SQ8.
5. **Integrated Query Pipeline**: Multi-stage execution pipeline routing from Compass orient/peel through intra-Locule graph search to exact FP32 Seed Kernel reranking.

---

## 2. Upstream Vendoring & Algorithmic Authority

The core HNSW index is implemented directly on top of the reference implementation of `nmslib/hnswlib`:
- **Repository**: [https://github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib)
- **Commit**: `058d7a866e462c00f0a6dea8660969379d5916bd`
- **Vendored Headers** (in `third_party/hnswlib/`):
  - `hnswlib.h`: Core definitions and platform SIMD macros.
  - `hnswalg.h`: The reference `HierarchicalNSW` graph algorithm.
  - `space_l2.h`: Euclidean distance interface.
  - `space_ip.h`: Inner product distance interface.
  - `visited_list_pool.h`: Fast bitset-based visited list pool for beam search.
  - `stop_condition.h`: Heuristic early-termination predicates.
  - `tuple_queue.h`: Min/max priority queues for candidate tracking.

### 2.1 PIMPL Encapsulation

To prevent upstream template headers, exception types, and macros from leaking into PomaiDB's public C++ and C API interfaces, `HnswIndex` is implemented using the **Pointer to Implementation (PIMPL)** idiom:

```cpp
// src/hnsw_index.h
class HnswIndex {
public:
    explicit HnswIndex(uint32_t dim, HnswOptions opts = {}, MetricType metric = MetricType::kL2);
    ~HnswIndex();
    
    Status Add(VectorId id, std::span<const float> vec);
    Status Search(std::span<const float> query, uint32_t topk, int ef_search,
                  std::vector<VectorId>* out_ids, std::vector<float>* out_dists,
                  IdFilter* filter = nullptr) const;
    Status SaveToBuffer(std::vector<uint8_t>* out) const;
    Status LoadFromBuffer(const uint8_t* data, size_t size);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

### 2.2 Native SIMD Kernel Bridge (`PomaiDistanceSpace`)

Rather than relying on generic fallback arithmetic, `PomaiDistanceSpace` binds `hnswlib::SpaceInterface<float>` directly to PomaiDB's AVX2-accelerated math kernels:
- **L2 Distance**: Evaluated via `pomai::core::L2Sq(a, b)` using 256-bit FMA vector registers (`vfmadd231ps`).
- **Inner Product**: Evaluated via `pomai::core::Dot(a, b)`. Distance returned as $1.0 - \text{Dot}(a, b)$ so that minimizing distance maximizes dot product.
- **Cosine Distance**: Evaluated via `pomai::core::CosineDistance(a, b) = 1.0 - \text{CosineSimilarity}(a, b)` with safe handling of zero-norm vectors.

---

## 3. Phase 1: Lock-Free Rind Read Concurrency

### 3.1 Problem Analysis
In earlier revisions, when `PomegranateQuery::Execute` scanned live MemTables in the uncompacted Rind layer, it evaluated vector validity by calling `Rind::IsDeleted(id)`. Internally, this acquired a reader-writer lock or mutex on the tombstone set for every single candidate vector, causing read concurrency to bottleneck and scale negatively with thread count.

### 3.2 Solution: Point-in-Time Tombstone Snapshot
PomaiDB introduced `RindTombstoneSnapshot`:
- When a query begins execution, it takes a single atomic shared-pointer copy of the tombstone hash set via `rind->CaptureTombstoneSnapshot()`.
- During the entire inner scan loop across both Rind and Locule Arils, vector IDs are verified against this immutable point-in-time snapshot.
- Zero mutexes or reader locks are acquired in inner loops, yielding lock-free query execution.

---

## 4. Phase 2: Spatial Locule Partitioning in Press

### 4.1 Problem Analysis
Previously, `Press::Compact` grouped vectors sequentially into Locules ordered by `VectorId`. Because ID order has zero correlation with vector spatial coordinates, every Locule's bounding sphere encompassed nearly the entire coordinate space ($r \approx \text{max\_dist}$). Consequently, Compass routing could prune zero candidate Locules.

### 4.2 Balanced Spatial K-Means
In `src/press.cc`, `PartitionItemsSpatially` implements balanced K-Means clustering prior to Locule generation:
1. **Cluster Count Estimation**:
   $$K = \max\left(1, \left\lceil \frac{N}{\text{locule\_capacity}} \right\rceil\right)$$
2. **K-Means++ Centroid Seeding**: Initial centroids are seeded using $D^2$ probability sampling to maximize separation across high-dimensional space.
3. **Constrained Assignment**: Vectors are assigned to their nearest cluster centroid subject to maximum capacity limits ($\le 1.2 \times \text{locule\_capacity}$) using Euclidean distance (for L2) or Spherical cosine similarity (for Cosine/Dot).
4. **Centroid and Radius Calculation**:
   $$\text{centroid} = \frac{1}{|C_k|} \sum_{v \in C_k} v, \quad \text{radius} = \max_{v \in C_k} \|v - \text{centroid}\|$$
This guarantees that Locules represent distinct, compact spatial neighborhoods with non-overlapping bounding spheres.

---

## 5. Phase 3 & 4: Zero-Copy Persistence in `.pom` Files

### 5.1 Graph Serialization in Press
During compaction, `ArilBuilder::Build` constructs an `HnswIndex` over the vectors belonging to each Aril, serializes the upstream `hnswlib` graph payload, prefixes it with a 32-byte `PomaiHnswHeader`, and writes it into the Aril file stream aligned to 64 bytes.

### 5.2 Zero-Copy Stream Deserialization
Upstream `hnswlib::HierarchicalNSW::loadIndexNoExceptions` requires an `std::istream` where offset 0 represents the beginning of the index. When opening a `.pom` file via zero-copy `mmap`:
- PomaiDB implements `imemstream`, a custom stream backed by `membuf` that directly overrides `seekoff` and `seekpos`.
- The buffer is mapped to `[base_addr + aril_hdr.graph_offset + sizeof(PomaiHnswHeader), graph_payload_size]`.
- Upstream `hnswlib` deserializes its graph tables directly from the memory-mapped file pages without memory duplication.

### 5.3 CRC32C Integrity & Graceful Pulp SQ8 Fallback
Before loading the graph:
1. `PomaiHnswHeader.magic` is checked against `0x484E5357` (`'HNSW'`).
2. `PomaiHnswHeader.checksum` is verified against `pomai::util::Crc32c` of the graph bytes.
3. If corruption is detected, `ArilReader` logs a warning and leaves `reader->graph_ = nullptr`.
4. `PomegranateQuery::Execute` detects `aril->HasGraph() == false` and falls back to SIMD-accelerated Pulp SQ8 flat scan, ensuring zero queries fail.

---

## 6. Phase 5: Production Integrated Query Pipeline

```text
               User Query Vector
                      │
                      ▼
             Compass Orient & Peel
        (Rank Locules by Centroid Dist)
                      │
                      ▼
         For each Candidate Locule:
    ┌──────────────────────────────────┐
    │  Dynamic Spatial Peel Check:     │
    │  Is min_possible_dist > worst?   │
    │        ├── YES: Break / Prune    │
    │        └── NO:  Search Locule    │
    └─────────────────┬────────────────┘
                      │
                      ▼
            Intra-Locule HNSW Search
        (with ArilFilter for tombstones)
                      │
                      ├── If Graph Missing/Corrupt:
                      │   Fallback to Pulp SQ8 Flat Scan
                      ▼
            Bounded Candidate Heap
            (Deduplicated Vector IDs)
                      │
                      ▼
          Exact FP32 Seed Kernel Rerank
             (via SeedKernel Vectors)
                      │
                      ▼
            Deterministic Top-K Sort
                      │
                      ▼
               Search Results
```

### 6.1 In-Graph Filtering (`ArilFilter`)
Upstream `hnswlib` supports custom `BaseFilterFunctor` callbacks. PomaiDB attaches `ArilFilter`, which executes during graph traversal:
- Checks Aril Seed Scar (`scar.IsDeleted(slot)`).
- Checks live Rind tombstones (`rind_snap.IsDeleted(id)`).
- Checks user metadata filters (`device_id`, `location_id`, `timestamp`, `lsn`).
This ensures non-qualifying nodes are pruned during graph traversal without polluting candidate heaps.

### 6.2 Dynamic Spatial Peel
Because candidate Locules are pre-sorted by proximity to centroid:
- As soon as the candidate heap is full ($|H| \ge \text{pick\_target}$), the current worst score $S_{\text{worst}}$ is known.
- For all subsequent candidate Locules, if their lower-bound distance $D_{\text{min}} > S_{\text{worst}}$, they are mathematically guaranteed to contain zero better vectors.
- The loop terminates immediately (`break;`), pruning distant Locules with zero search overhead.

### 6.3 Exact FP32 Seed Kernel Reranking
Candidates selected by HNSW or Pulp are reranked against their true FP32 vectors stored in the Seed Kernel:
- Evaluated via `pomai::core::ComputeMetricScore(metric, query, span)`.
- Top-$K$ items are extracted via `pomai::core::SelectTopK` using deterministic tie-breaking (scores first, lower `VectorId` second).
