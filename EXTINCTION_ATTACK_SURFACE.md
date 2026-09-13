# PomaiDB — EXTINCTION ATTACK SURFACE

Generated: 2026-09-13
Campaign: Catastrophic Adversarial Destruction and Reliability Audit (Rounds 1-4)

---

## Scope

This document enumerates every attack surface probed during the Extinction Event campaign.

---

## 1. Persistence Layer — Locule File Format

### 1.1 Locule::Open — File-Mapped Container Reader

ATTACK VECTOR                                             | RESULT
Invalid magic                                             | REJECTED (safe)
Invalid version                                           | REJECTED (safe)
directory_size larger than file_size -> OOB Crc32c call   | BUG FIXED
directory_size less than required_dir_size                | BUG FIXED
directory_offset + directory_size integer overflow        | BUG FIXED
aril_offset + aril_size > file_size                       | FIXED (prior round)
aril_offset + aril_size integer overflow                  | FIXED (prior round)
Per-entry checksum mismatch                               | REJECTED (safe)
Footer offset truncation                                  | REJECTED (safe)
Footer magic mismatch                                     | REJECTED (safe)
Centroid size overflow                                    | REJECTED (safe)
Bit-flip in header fields (10,000 mutations)              | ALL CRASH-FREE
Multi-byte stomp in header (10,000 mutations)             | ALL CRASH-FREE
Extreme 64-bit integer injection (10,000 mutations)       | ALL CRASH-FREE

### 1.2 Locule::OpenFromMemory — In-Memory Locule Reader

ATTACK VECTOR                                             | RESULT
Missing version check                                     | BUG FIXED
directory_size OOB before Crc32c                          | BUG FIXED
Per-entry aril_offset >= file_size_ not caught            | BUG FIXED
Per-entry aril_size > file_size_ not caught               | BUG FIXED
Per-entry integer overflow not caught                     | BUG FIXED
Per-entry checksum not validated                          | BUG FIXED

### 1.3 ArilReader::OpenFromMemory — Aril Section Reader

ATTACK VECTOR                                             | RESULT
dimension == 0 not validated                              | BUG FIXED
dimension > 65536 not validated                           | BUG FIXED
vector_count > max_size not validated                     | BUG FIXED
Pulp section size vs declared vector_count inconsistency  | BUG FIXED
Section offset/size OOB (kernel, graph, scar, dir, meta)  | FIXED (prior round)
Section overflow (off + size < off)                       | FIXED (prior round)

---

## 2. Index Layer — HNSW Persistence

### 2.1 HnswIndex::LoadFromBuffer — Graph Deserialization

ATTACK VECTOR                                             | RESULT
[P0] Metric not restored after deserialization            | BUG FIXED
Magic/version mismatch in header                          | REJECTED (safe)
Dimension mismatch                                        | REJECTED (safe)
Truncated buffer                                          | REJECTED (safe)
Corrupt graph data                                        | GRACEFUL FALLBACK

CRITICAL BUG DETAIL:
LoadFromBuffer read metric_ from the serialized PomaiHnswHeader into the impl
field, but DID NOT update PomaiDistanceSpace::space_ distance function pointer.
Any locule built with InnerProduct or Cosine metric was silently searched with
L2 after a reopen/rehydration cycle => 0.26 Recall@10 instead of 0.99.

FIX: Added SetMetric() to PomaiDistanceSpace, called from LoadFromBuffer.

---

## 3. Query Layer — Spatial Routing and Candidate Pruning

### 3.1 Compass::Orient

ATTACK VECTOR                                             | RESULT
Centroid dimension mismatch (no panic)                    | SAFE
Dot/cosine upper bound wrong (dot + radius vs dot+||q||r) | BUG FIXED (Cauchy-Schwarz)
Pruning with break instead of continue (prior session)    | FIXED

### 3.2 HNSW ef_search Starvation

ef_search capped at 64 starving multi-cluster segments    | FIXED (prior session)

---

## 4. Stateful Fuzzer Coverage

Operations: INSERT, UPSERT, DELETE, QUERY, FILTERED_QUERY, BATCH_INSERT,
            BATCH_DELETE, COMPACT, FLUSH, REOPEN

Seed         | Operations | Mean Recall@10
42           | 20,000     | 99.29%
1337         | 20,000     | 99.70%
2026         | 20,000     | 99.66%
0xDEADBEEF  | 20,000     | 99.51%

---

## 5. Crash-Recovery Campaign

- 100 power-loss / torn-WAL cycles
- All 100 cycles: zero data loss, zero corruptions
- WAL replay verified correct after every forced truncation

---

## 6. Concurrency Extinction

- 8 concurrent readers, 4 concurrent writers, 2 deleters, 2 compactors
- 6,757 queries | 49,793 writes | 75,631 deletes | 6 compactions
- Zero deadlocks, zero panics, zero data races detected

---

## 7. Persistence Mutation War

- 10,000 bit/byte/block mutations on real .pom locule files
- 9,462 rejected (corrupted as expected)
- 538 accepted (structurally valid - none crashed during query)
- Zero segfaults across all 10,000 mutations
