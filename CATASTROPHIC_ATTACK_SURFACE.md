# PomaiDB — Catastrophic Attack Surface & Threat Model

## 1. Complete Database Lifecycle Map

```text
                  [ User / Client Application ]
                                │
                                ▼
                     pomai::DB::Open(opt)
                                │
                                ▼
         MembraneManager::Open() ───► storage::Manifest::ListMembranes()
                                │
                                ▼
                   VectorEngine::Open()
                                │
                                ▼
                 PomegranateEngine::Open()
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
     ingest::Rind::Open()                manifest::FruitMap::Open()
             │                                     │
             ├─► wal.log (WAL file)                ├─► manifest.json
             └─► Active MemTable (RAM)             └─► Locules (.pom files)
                                                          │
                                                          ▼
                                                    storage::Locule::Open()
                                                          │
                                                          ▼
                                                    storage::ArilReader::OpenFromMemory()
                                                          │
                                                          ├─► PulpView (SQ8)
                                                          ├─► SeedKernelView (FP32)
                                                          ├─► SeedScarView (Tombstones)
                                                          └─► HnswIndex::LoadFromBuffer()
```

### Mutational & Query Lifecycle

```text
Insert / Put:
  DB::Put() ──► MembraneManager ──► VectorEngine ──► PomegranateEngine ──► Rind::Put()
                                                                              ├─► WAL Append
                                                                              └─► Active MemTable (skiplist/hash)

Delete:
  DB::Delete() ──► Rind::Delete()
                        ├─► WAL Append (Op::kDel)
                        ├─► Active MemTable tombstone
                        └─► Atomic Tombstones set insertion

Compaction / Press:
  DB::Compact() ──► PomegranateEngine::Compact()
                          │
                          ├─► Rind::Freeze() (Active MemTable -> Frozen MemTable)
                          │
                          └─► Press::Compact(rind, fruit_map)
                                    │
                                    ├─► Extract live items from existing Locules (Drying)
                                    ├─► Merge frozen MemTables (overwriting / deleting)
                                    ├─► Reseeding: PartitionItemsSpatially (Balanced K-Means)
                                    ├─► Pressing: For each partition, construct Arils:
                                    │         ├─► Pulp (SQ8 Quantization)
                                    │         ├─► Seed Kernel (Exact FP32)
                                    │         ├─► Seed Scar (Tombstone bitset)
                                    │         └─► HnswIndex (Upstream hnswlib construction & buffer save)
                                    ├─► Write .pom file (PomaiFileHeader + Arils + ArilDirectory + LoculeFooter)
                                    ├─► Update FruitMap manifest
                                    ├─► Atomic Snapshot swap
                                    └─► Delete obsolete .pom files

Query / Search:
  DB::Search() ──► PomegranateEngine::Search()
                          │
                          ├─► Capture RindTombstoneSnapshot (lock-free snapshot)
                          ├─► Rind::Taste() (Brute-force scan on live active/frozen memtables)
                          ├─► Compass::Orient() (Sort candidate Locules by centroid distance)
                          ├─► Dynamic Spatial Peel (Prune candidate Locules whose lower bound > heap worst)
                          ├─► For each surviving Locule:
                          │         └─► For each Aril:
                          │                   ├─► HnswIndex::Search(query, topk, ef_search, &ArilFilter)
                          │                   └─► Fallback on error/corruption: SIMD Pulp SQ8 flat scan
                          ├─► Global Pick Heap merge & deduplication
                          ├─► Exact FP32 Seed Kernel reranking (ComputeMetricScore)
                          └─► Deterministic Top-K sort (SelectTopK)
```

---

## 2. Attack Surface Boundaries & Vulnerability Vectors

### 2.1 Persistence & Binary Serialization Boundaries

| File | Structure / Function | Attack Vectors & Boundary Risks |
|---|---|---|
| [`src/locule.cc`](file:///e:/GithubProjects/pomaidb/src/locule.cc) | `Locule::Open` | `hdr.directory_offset + required_dir_size` integer overflow; out-of-bounds read; malformed `footer_offset`; directory checksum tampering. |
| [`src/locule.cc`](file:///e:/GithubProjects/pomaidb/src/locule.cc) | `Locule::Open` | `entry.aril_offset + entry.aril_size` integer overflow wrapping past `file_size_`; crash in `Crc32c`. |
| [`src/aril.cc`](file:///e:/GithubProjects/pomaidb/src/aril.cc) | `validate_bounds` | `off + size > max_size`: 64-bit unsigned integer wraparound if `off + size < off`. Out-of-bounds pointer calculation. |
| [`src/aril.cc`](file:///e:/GithubProjects/pomaidb/src/aril.cc) | `ArilReader::GetMetadata` | `offsets[slot] >= offsets[slot + 1]` or malformed string lengths reading beyond `meta_size_`. |
| [`src/hnsw_index.cc`](file:///e:/GithubProjects/pomaidb/src/hnsw_index.cc) | `HnswIndex::LoadFromBuffer` | Header truncation (`len < sizeof(PomaiHnswHeader)`); dimension mismatch; corrupted `hnswlib` node count; corrupted enterpoint node. |
| [`src/wal.cc`](file:///e:/GithubProjects/pomaidb/src/wal.cc) | `Wal::Replay` | Frame truncation; frame length overflow; corrupted CRC; metadata string length out of bounds; encrypted frame tampering. |
| [`src/fruit_map.cc`](file:///e:/GithubProjects/pomaidb/src/fruit_map.cc) | `FruitMap::LoadManifest` | JSON corruption; truncated manifest file; manifest referencing missing `.pom` files. |

### 2.2 Memory Safety & Casting Boundaries

| Code Location | Expression / Cast | Risk / Threat |
|---|---|---|
| [`src/aril.cc:71`](file:///e:/GithubProjects/pomaidb/src/aril.cc#L71) | `reinterpret_cast<const float*>(base_addr + hdr.kernel_offset)` | Unaligned float read if `kernel_offset` not aligned to 4/32/64 bytes; OOB read if `kernel_size < vector_count * dim * 4`. |
| [`src/aril.cc:82`](file:///e:/GithubProjects/pomaidb/src/aril.cc#L82) | `reinterpret_cast<const format::SeedDirectoryEntry*>` | Memory boundary violation if `dir_size < vector_count * sizeof(SeedDirectoryEntry)`. |
| [`src/hnsw_index.cc:110`](file:///e:/GithubProjects/pomaidb/src/hnsw_index.cc#L110) | `reinterpret_cast<const float*>(a)` | Buffer over-read in AVX2 distance kernels if vector data buffer in `hnswlib` is not aligned or smaller than `dim * 4`. |
| [`src/hnsw_index.cc:54`](file:///e:/GithubProjects/pomaidb/src/hnsw_index.cc#L54) | `membuf` / `imemstream` | Dangling pointer if underlying `mmap` is closed while an `imemstream` or `HnswIndex` is alive. |
| [`src/locule.cc:32`](file:///e:/GithubProjects/pomaidb/src/locule.cc#L32) | `FileMapping` / `mmap` | Windows file lock contention preventing deletion of superseded generation `.pom` files during compaction. |

### 2.3 Thread-Safety & Concurrency Boundaries

| Component | Synchronization Mechanism | Threat / Potential Race |
|---|---|---|
| `Rind` | `std::shared_mutex rw_lock_` | Concurrent `Put` / `Delete` vs `CaptureTombstoneSnapshot` vs `Taste` vs `Freeze`. |
| `PomegranateQuery` | Lock-free via `RindTombstoneSnapshot` | Snapshot invalidation or UAF if shared pointer life cycle is mishandled. |
| `FruitMap` | `std::mutex mutex_` | Race between `InstallSnapshot` (during Press) and concurrent queries holding `CurrentSnapshot()`. |
| `Press` | Single-threaded compaction loop | Race if `TakeFrozenMemtables()` is called concurrently with MemTable mutation. |
| `MembraneManager` | `std::shared_mutex mu_` | Concurrent `OpenMembrane` / `DropMembrane` / `Compact` while queries are executing. |

---

## 3. Threat Matrix & Attack Plan

1. **Integer Overflow & OOB Memory Exploits**:
   Inject UINT64_MAX offsets, corrupt Aril sections, truncate files at arbitrary byte boundaries.
2. **Crash Inconsistency & Power Loss Simulation**:
   Simulate sudden process termination during Press, WAL write, and Manifest commit.
3. **Compass Pruning Boundary Attacks**:
   Craft datasets where true nearest neighbors lie on Locule edges to trigger false pruning.
4. **HNSW Pathological Geometry Attacks**:
   100% duplicate vectors, zero vectors, antipodal pairs, extreme scales ($10^{-30}$ to $10^{30}$).
5. **Concurrent Mutation Destruction**:
   Multithreaded barrage of concurrent inserts, deletes, queries, compactions, and closes.
