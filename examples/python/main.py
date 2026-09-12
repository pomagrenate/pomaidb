#!/usr/bin/env python3
"""
PomaiDB Python Comprehensive Feature Pipeline Example
=====================================================
Demonstrates 100% of PomaiDB's capabilities:
  1. Engine configuration (Quantization: SQ8, Metric: Cosine, Memory budget)
  2. Database lifecycle (Open, Close, RAII context manager)
  3. Multi-membrane management (Create, Open, Close, List, Drop)
  4. Vector CRUD (Put, Put Batch, Get, Exists, Delete)
  5. Arbitrary payloads & event timestamps
  6. Top-K ANN vector search
  7. Membrane-scoped vector search
  8. Point-in-time temporal queries (as_of_ts)
  9. Zero-copy query session
 10. Maintenance operations (Flush, Freeze, Compact)
 11. Engine telemetry and statistics (get_stats)
"""

import os
import shutil
import time
import json
from pomaidb import (
    Database,
    QUANT_SQ8,
    QUANT_NONE,
    open_db,
    close,
    put,
    get,
    exists,
    delete,
    search,
    search_zero_copy,
    release_zero_copy_session,
    flush,
    freeze,
    compact,
    create_membrane,
    drop_membrane,
    list_membranes,
    get_stats,
)

DB_PATH = "./pomaidb_python_example_store"
DIMENSION = 64

def generate_vector(seed_val: float) -> list[float]:
    """Generate a normalized test vector."""
    vec = [(seed_val + i * 0.01) for i in range(DIMENSION)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]

def main():
    print("=" * 70)
    print(" PomaiDB Python Comprehensive Feature Pipeline")
    print("=" * 70)

    # Clean previous run if present
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # -------------------------------------------------------------------------
    # 1. DATABASE INITIALIZATION & CONFIGURATION
    # -------------------------------------------------------------------------
    print("\n[Step 1] Opening PomaiDB with SQ8 Quantization & Cosine Metric...")
    db = Database.open(
        path=DB_PATH,
        dim=DIMENSION,
        shards=2,
        metric="cosine",
        quant_type=QUANT_SQ8,
        memory_budget_bytes=64 * 1024 * 1024, # 64MB memory budget
        auto_freeze_on_pressure=True,
        memtable_flush_threshold_mb=16,
    )
    print(f"  -> Database opened at: {DB_PATH}")

    # -------------------------------------------------------------------------
    # 2. MULTI-MEMBRANE LIFECYCLE
    # -------------------------------------------------------------------------
    print("\n[Step 2] Multi-Membrane Architecture Management...")
    print("  -> Creating 'realtime_events' membrane (dim=64, shards=2)...")
    db.create_membrane("realtime_events", dim=DIMENSION, shard_count=2)

    print("  -> Creating 'archival_docs' membrane (dim=64, shards=1)...")
    db.create_membrane("archival_docs", dim=DIMENSION, shard_count=1)

    membranes = db.list_membranes()
    print(f"  -> Active membranes: {membranes}")

    # -------------------------------------------------------------------------
    # 3. VECTOR CRUD & PAYLOAD INSERTION
    # -------------------------------------------------------------------------
    print("\n[Step 3] Vector Insertion with Payloads and Timestamps...")
    now_ts = int(time.time() * 1000)

    # Put into default membrane
    vec1 = generate_vector(1.0)
    payload1 = json.dumps({"user": "alice", "tag": "login"}).encode("utf-8")
    db.put(id=101, vector=vec1, timestamp=now_ts, payload=payload1)
    print(f"  -> Inserted ID 101 into default membrane with timestamp {now_ts}")

    # Put into 'realtime_events' membrane
    vec2 = generate_vector(2.0)
    payload2 = json.dumps({"sensor": "lidar_front", "speed_kmh": 65.4}).encode("utf-8")
    db.put(id=201, vector=vec2, membrane="realtime_events", timestamp=now_ts + 10, payload=payload2)
    print(f"  -> Inserted ID 201 into 'realtime_events' membrane with payload")

    # Batch insertion into 'archival_docs'
    batch_ids = [301, 302, 303]
    batch_vectors = [generate_vector(3.0 + i) for i in range(3)]
    db.put_batch(ids=batch_ids, vectors=batch_vectors, membrane="archival_docs")
    print(f"  -> Batch inserted IDs {batch_ids} into 'archival_docs' membrane")

    # -------------------------------------------------------------------------
    # 4. EXISTENCE & RECORD RETRIEVAL
    # -------------------------------------------------------------------------
    print("\n[Step 4] Checking Existence and Retrieving Full Records...")
    exists_101 = db.exists(101)
    exists_201_memb = db.exists(201, membrane="realtime_events")
    print(f"  -> Exists 101 (default): {exists_101}")
    print(f"  -> Exists 201 (realtime_events): {exists_201_memb}")

    # Get record with metadata, payload, timestamp
    rec = db.get(201, membrane="realtime_events", with_metadata=True)
    if rec:
        print(f"  -> Retrieved Record 201:")
        print(f"     Dim: {rec.dim}")
        print(f"     Timestamp: {rec.timestamp}")
        print(f"     Payload: {rec.payload.decode('utf-8')}")
        print(f"     Vector snippet: {[round(x, 4) for x in rec.vector[:4]]}...")

    # -------------------------------------------------------------------------
    # 5. VECTOR SEARCH (ANN) & MEMBRANE-SCOPED SEARCH
    # -------------------------------------------------------------------------
    print("\n[Step 5] Top-K Vector Search...")
    query_vec = generate_vector(1.05) # Closest to ID 101

    # Search in default membrane
    hits = db.search(query_vec, topk=3)
    print(f"  -> Search in default membrane (Top 3):")
    for hit in hits:
        print(f"     Hit ID: {hit.id}, Cosine Score: {hit.score:.4f}")

    # Scoped search in 'archival_docs'
    query_arch = generate_vector(3.1)
    arch_hits = db.search(query_arch, topk=3, membrane="archival_docs")
    print(f"  -> Scoped Search in 'archival_docs' membrane:")
    for hit in arch_hits:
        print(f"     Hit ID: {hit.id}, Cosine Score: {hit.score:.4f}")

    # Batch search
    batch_queries = [generate_vector(3.0), generate_vector(3.2)]
    batch_results = db.search_batch(batch_queries, topk=2, membrane="archival_docs")
    print(f"  -> Batch Search returned {len(batch_results)} query result sets")

    # -------------------------------------------------------------------------
    # 6. TEMPORAL TIME-TRAVEL QUERY
    # -------------------------------------------------------------------------
    print("\n[Step 6] Temporal Query (Point-in-Time as_of_ts)...")
    past_hits = db.search(query_vec, topk=3, as_of_ts=now_ts + 5)
    print(f"  -> Search as_of_ts={now_ts + 5}: Found {len(past_hits)} hits")

    # -------------------------------------------------------------------------
    # 7. ZERO-COPY QUERY SESSION
    # -------------------------------------------------------------------------
    print("\n[Step 7] Zero-Copy Memory Bridge Query...")
    zc_res = search_zero_copy(db._handle, query_vec, topk=2)
    print(f"  -> Zero-copy hits: {zc_res['hits']}")
    print(f"  -> Memory session ID: {zc_res['session_id']}")
    if zc_res["session_id"]:
        release_zero_copy_session(zc_res["session_id"])
        print("  -> Released zero-copy memory session cleanly")

    # -------------------------------------------------------------------------
    # 8. ENGINE MAINTENANCE (FLUSH, FREEZE, COMPACT)
    # -------------------------------------------------------------------------
    print("\n[Step 8] Engine Maintenance Operations...")
    print("  -> Flushing memtable to disk...")
    db.flush()

    print("  -> Freezing active write membrane...")
    db.freeze("realtime_events")

    print("  -> Triggering background compaction...")
    db.compact("realtime_events")
    print("  -> Maintenance cycle complete.")

    # -------------------------------------------------------------------------
    # 9. ENGINE TELEMETRY & STATS
    # -------------------------------------------------------------------------
    print("\n[Step 9] Fetching Engine Health & Statistics...")
    stats = db.get_stats()
    print(f"  -> Version: {stats.get('version')}")
    print(f"  -> ABI Version: {stats.get('abi_version')}")
    print(f"  -> Active Membranes: {stats.get('membranes')}")

    # -------------------------------------------------------------------------
    # 10. DELETION & CLEANUP
    # -------------------------------------------------------------------------
    print("\n[Step 10] Deletion & Clean Shutdown...")
    db.delete(101)
    print(f"  -> Deleted ID 101. Exists now: {db.exists(101)}")

    db.drop_membrane("archival_docs")
    print(f"  -> Dropped 'archival_docs'. Current membranes: {db.list_membranes()}")

    db.close()
    print("  -> Database closed successfully.")

    # Clean up test directory
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("\n" + "=" * 70)
    print(" [SUCCESS] All PomaiDB Python pipeline features executed cleanly!")
    print("=" * 70)

if __name__ == "__main__":
    main()
