# PomaiDB — EXTINCTION FAILURE CORPUS

Generated: 2026-09-13
Campaign: Extinction Event — All confirmed real bugs with reproduction steps

---

## FAILURE-001: HNSW Metric Deserialization Bug [P0]

### Classification
- Severity: P0 — Silent Correctness Failure
- Component: src/hnsw_index.cc
- Discovered by: recall_test (Recall@10 = 0.26)

### Symptom
Any PomegranateEngine opened from disk with InnerProduct or Cosine metric
reports Recall@10 = 0.26 instead of 0.99+. The failure is silent — no
error codes, no warnings, queries complete successfully with wrong results.

### Root Cause
HnswIndexImpl::LoadFromBuffer() at line ~303 wrote:
  metric_ = static_cast<MetricType>(hdr.metric);
but PomaiDistanceSpace space_ (which contains the actual distance function
pointer used by hnswlib during graph traversal) was initialized in the
constructor with the default metric and NEVER updated. So every graph
search after deserialization used L2 even if the index was built with IP.

### Minimal Reproduction
  DBOptions opt; opt.metric = MetricType::kInnerProduct;
  engine.Open(); engine.Put(id, vec); engine.Compact(); engine.Close();
  engine.Open();  // re-open from disk
  engine.Search(query, k, &results);
  // Recall@10 ~ 0.26 instead of 0.99

### Fix
  src/hnsw_index.cc — Added SetMetric() to PomaiDistanceSpace:
    void SetMetric(MetricType m) {
      metric_ = m; switch (m) { ... update dist_func_ ... }
    }
  Called space_.SetMetric(metric_) in LoadFromBuffer() after restoring metric_.

---

## FAILURE-002: OOB Crc32c in Locule::Open [P1]

### Classification
- Severity: P1 — Segfault on Corrupt Input
- Component: src/locule.cc (Open)
- Discovered by: persistence_mutation_war (0xC0000005 crash)

### Symptom
Corrupted hdr.directory_size larger than file_size_ causes Crc32c to read
beyond the mapped memory region, producing access violation (SIGSEGV /
0xC0000005 on Windows).

### Root Cause
Code at line ~99:
  if (hdr.checksum != 0 && hdr.directory_size > 0) {
      uint32_t dir_crc = Crc32c(base_addr + hdr.directory_offset,
                                 hdr.directory_size);  // OOB!
  }
hdr.directory_size was read directly from the (potentially corrupt) file
header without being validated against file_size_ or required_dir_size.

### Fix
Added 4-condition bounds check before the Crc32c call:
  if (hdr.directory_size < required_dir_size ||
      hdr.directory_size > locule->file_size_ ||
      hdr.directory_offset + hdr.directory_size > locule->file_size_ ||
      hdr.directory_offset + hdr.directory_size < hdr.directory_offset) {
      return Status::Corruption("header directory size invalid");
  }

---

## FAILURE-003: Missing Validation in Locule::OpenFromMemory [P1]

### Classification
- Severity: P1 — Segfault + Silent Corruption Risk
- Component: src/locule.cc (OpenFromMemory)
- Discovered by: persistence_mutation_war

### Root Cause
OpenFromMemory lacked:
1. Version check (header version field ignored)
2. directory_size bounds check before Crc32c (same as FAILURE-002)
3. Per-entry aril_offset >= file_size_ check (only aril_offset+size was checked)
4. Per-entry integer overflow check (aril_offset + aril_size < aril_offset)
5. Per-entry checksum validation (entry.checksum not validated)

### Fix
Full parity with Locule::Open applied:
- Version check added
- 4-condition directory_size bounds check added
- Per-entry 4-condition bounds + overflow check added
- Per-entry Crc32c validation added

---

## FAILURE-004: ArilReader Missing Dimension Guards [P2]

### Classification
- Severity: P2 — Potential Crash / Garbage Data
- Component: src/aril.cc (OpenFromMemory)
- Discovered by: persistence_mutation_war

### Root Cause
Missing guards:
- hdr.dimension == 0 allowed divide-by-zero / zero-size allocation paths
- hdr.dimension > 65536 allowed absurdly large allocations
- hdr.vector_count > max_size allowed impossible declared counts
- hdr.pulp_size < vector_count * dimension not checked

### Fix
Added guards after validate_bounds calls in OpenFromMemory.

---

## FAILURE-005: Cauchy-Schwarz Upper Bound Wrong [P2]

### Classification
- Severity: P2 — Recall Degradation Under Non-Unit Queries
- Component: src/compass.cc (Compass::Orient)
- Discovered by: Code review during extinction audit

### Root Cause
For dot/cosine metric, the upper bound on the score achievable by any point
in a locule (centroid c, radius r) given query q is:
  <q, c> + ||q|| * r      (Cauchy-Schwarz)
but the code computed:
  ol.min_possible_distance = dot + radius;   // WRONG: missing ||q||
This made the upper bound too tight, causing aggressive over-pruning of
locules when ||q|| != 1 (unnormalized queries). Recall degradation is
proportional to the deviation of ||q|| from 1.

### Fix
Pre-compute q_norm before the locule loop:
  float q_norm = sqrt(sum(q[i]^2))
Use q_norm * radius instead of radius in the bound.

---

## FAILURE-006: Premature Locule Pruning [P1] (Prior Session)

### Classification
- Severity: P1 — Recall Degradation
- Component: src/pomegranate_query.cc
- Fixed: Prior session

### Root Cause
reak instead of continue in the spatial pruning loop meant that as
soon as one locule could be pruned, ALL remaining locules were skipped.

### Fix
Changed reak to continue.

---

## FAILURE-007: ef_search Starvation [P2] (Prior Session)

### Classification
- Severity: P2 — Recall Degradation on Large Segments
- Component: src/pomegranate_query.cc
- Fixed: Prior session

### Root Cause
ef_search capped at 64, insufficient for large multi-cluster arils.

### Fix
Scaled: ef_search = max(opts.ef_search, aril_k * 6, 512)
