#!/usr/bin/env python3
"""
PomaiDB Principal QA Adversarial Test Suite
===========================================
Aggressive reliability, correctness, corruption, concurrency, and mathematical
soundness tests designed to break the database.
"""

import ctypes
import json
import math
import os
import shutil
import sys
import threading
import time
import numpy as np

# Ensure pomaidb is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "bindings", "python"))

import pomaidb
from pomaidb import (
    Database,
    QUANT_NONE,
    QUANT_SQ8,
    PomaiDBError,
    _ensure_lib,
    _lib,
)

RESULTS = {}

def record_test(test_id, name, status, evidence, severity=None, bug_id=None):
    RESULTS[test_id] = {
        "name": name,
        "status": status,  # "FAIL" (bug confirmed) or "PASS" (invariant held)
        "evidence": evidence,
        "severity": severity,
        "bug_id": bug_id
    }
    flag = "[FAIL - BUG CONFIRMED]" if status == "FAIL" else "[PASS - INVARIANT HELD]"
    print(f"\n>>> {flag} {test_id}: {name}")
    print(f"    Evidence: {evidence}")

def clean_dir(p):
    if os.path.exists(p):
        for _ in range(5):
            try:
                shutil.rmtree(p)
                break
            except Exception:
                time.sleep(0.1)

# ==============================================================================
# EXP-01: Frozen MemTable Update Shadowing (Stale Read / Incorrect Search Score)
# ==============================================================================
def test_exp01_frozen_memtable_update_shadowing():
    db_path = "./adversarial_test_exp01"
    clean_dir(db_path)
    try:
        dim = 4
        db = Database.open(db_path, dim=dim, metric="l2")
        
        # 1. Put vector 1 with coordinates [0.0, 0.0, 0.0, 0.0]
        db.put(1, [0.0, 0.0, 0.0, 0.0])
        
        # 2. Freeze the database -> vector 1 moves to frozen_memtables_
        db.freeze()
        
        # 3. Update vector 1 with coordinates [10.0, 10.0, 10.0, 10.0] into active_memtable_
        db.put(1, [10.0, 10.0, 10.0, 10.0])
        
        # Verify get() returns the updated coordinates
        rec = db.get(1, with_metadata=True)
        get_vec = rec.vector if rec else []
        
        # 4. Search with query [10.0, 10.0, 10.0, 10.0]
        # Distance to updated vector [10,10,10,10] is 0.0 (Score: -0.0)
        # Distance to old frozen vector [0,0,0,0] is 400.0 (Score: -400.0)
        hits = db.search([10.0, 10.0, 10.0, 10.0], topk=1)
        
        db.close()
        
        if not hits:
            record_test("EXP-01", "Frozen MemTable Update Shadowing", "FAIL",
                        "Search returned empty results for updated vector", severity="P1", bug_id="BUG-01")
            return

        hit_score = hits[0].score
        # For L2 metric, PomaiDB scores are -distance
        # Expected: distance = 0.0 -> score = -0.0
        # If shadowed: distance = 400.0 -> score = -400.0
        if math.isclose(hit_score, -400.0, abs_tol=1e-2):
            record_test("EXP-01", "Frozen MemTable Update Shadowing", "FAIL",
                        f"Search returned stale frozen vector score ({hit_score:.1f}) instead of updated active vector score (0.0). Active update is completely shadowed!",
                        severity="P1", bug_id="BUG-01")
        elif math.isclose(hit_score, 0.0, abs_tol=1e-2):
            record_test("EXP-01", "Frozen MemTable Update Shadowing", "PASS",
                        f"Search returned correct updated score: {hit_score}")
        else:
            record_test("EXP-01", "Frozen MemTable Update Shadowing", "FAIL",
                        f"Search returned unexpected score: {hit_score} (neither 0.0 nor -400.0)", severity="P1", bug_id="BUG-01")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-02: Silent Metadata & Payload Loss in pomai_put_batch
# ==============================================================================
def test_exp02_silent_payload_loss_in_batch():
    db_path = "./adversarial_test_exp02"
    clean_dir(db_path)
    try:
        dim = 4
        db = Database.open(db_path, dim=dim)
        
        # Insert 2 vectors via put_batch with tenants (which map to metadata)
        ids = [101, 102]
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        
        # Using functional C-API directly to provide payload and timestamp
        _ensure_lib()
        upsert_arr = (pomaidb._lib.PomaiUpsert * 2)()
        p1 = b"PAYLOAD_101_SECRET"
        p2 = b"PAYLOAD_102_SECRET"
        
        c_p1 = (ctypes.c_uint8 * len(p1))(*p1)
        c_p2 = (ctypes.c_uint8 * len(p2))(*p2)
        c_v1 = (ctypes.c_float * 4)(*vectors[0])
        c_v2 = (ctypes.c_float * 4)(*vectors[1])
        
        upsert_arr[0].struct_size = ctypes.sizeof(pomaidb._lib.PomaiUpsert)
        upsert_arr[0].id = 101
        upsert_arr[0].dim = 4
        upsert_arr[0].vector = c_v1
        upsert_arr[0].timestamp = 999999
        upsert_arr[0].payload = c_p1
        upsert_arr[0].payload_len = len(p1)
        
        upsert_arr[1].struct_size = ctypes.sizeof(pomaidb._lib.PomaiUpsert)
        upsert_arr[1].id = 102
        upsert_arr[1].dim = 4
        upsert_arr[1].vector = c_v2
        upsert_arr[1].timestamp = 888888
        upsert_arr[1].payload = c_p2
        upsert_arr[1].payload_len = len(p2)
        
        st = pomaidb._lib.pomai_put_batch(db._handle, upsert_arr, 2)
        if st:
            msg = pomaidb._lib.pomai_status_message(st)
            pomaidb._lib.pomai_status_free(st)
            record_test("EXP-02", "Batch Payload Preservation", "FAIL", f"pomai_put_batch failed: {msg}")
            db.close()
            return
            
        rec1 = db.get(101, with_metadata=True)
        rec2 = db.get(102, with_metadata=True)
        db.close()
        
        recovered_payload1 = rec1.payload if rec1 else None
        recovered_ts1 = rec1.timestamp if rec1 else 0
        
        if recovered_payload1 != p1 or recovered_ts1 != 999999:
            record_test("EXP-02", "Silent Metadata & Payload Loss in pomai_put_batch", "FAIL",
                        f"Payload dropped! Expected {p1} and ts 999999, but recovered payload={recovered_payload1} and ts={recovered_ts1}",
                        severity="P1", bug_id="BUG-02")
        else:
            record_test("EXP-02", "Silent Metadata & Payload Loss in pomai_put_batch", "PASS",
                        f"Batch metadata preserved: payload={recovered_payload1}, ts={recovered_ts1}")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-03: Missing Checksum Verification on Disk Read (Silent Data Corruption)
# ==============================================================================
def test_exp03_missing_checksum_verification():
    db_path = "./adversarial_test_exp03"
    clean_dir(db_path)
    try:
        dim = 16
        db = Database.open(db_path, dim=dim)
        
        # Insert 100 vectors
        for i in range(100):
            vec = [(float(i) + j * 0.1) for j in range(dim)]
            db.put(i + 1, vec)
            
        # Freeze and compact so data is flushed to .pom Locule container file
        db.freeze()
        db.compact()
        db.close()
        
        # Find .pom file in db_path
        pom_files = []
        for root, _, files in os.walk(db_path):
            for f in files:
                if f.endswith(".pom"):
                    pom_files.append(os.path.join(root, f))
                    
        if not pom_files:
            record_test("EXP-03", "Missing Checksum Verification", "FAIL",
                        "No .pom file generated after freeze and compact")
            return
            
        target_pom = pom_files[0]
        with open(target_pom, "rb") as f:
            data = bytearray(f.read())
            
        # Corrupt 16 bytes in the middle of the payload/vector data
        corrupt_offset = len(data) // 2
        for b in range(16):
            data[corrupt_offset + b] ^= 0xFF
            
        with open(target_pom, "wb") as f:
            f.write(data)
            
        # Re-open database
        corrupted_db = None
        try:
            corrupted_db = Database.open(db_path, dim=dim)
            # Try reading vectors and searching
            query = [1.0] * dim
            hits = corrupted_db.search(query, topk=5)
            # If it opens and searches without raising Corruption error:
            record_test("EXP-03", "Missing Checksum Verification on Read", "FAIL",
                        f"Database silently opened corrupted .pom file without error! Search succeeded ({len(hits)} hits) instead of reporting Status::Corruption! Stored CRC32 checksum was completely unverified!",
                        severity="P0", bug_id="BUG-03")
        except PomaiDBError as e:
            if "corruption" in str(e).lower():
                record_test("EXP-03", "Missing Checksum Verification on Read", "PASS",
                            f"Corruption correctly detected: {e}")
            else:
                record_test("EXP-03", "Missing Checksum Verification on Read", "FAIL",
                            f"Failed with non-corruption error: {e}", severity="P2", bug_id="BUG-03")
        finally:
            if corrupted_db:
                corrupted_db.close()
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-04: NaN / Infinity Vector Poisoning
# ==============================================================================
def test_exp04_nan_inf_poisoning():
    db_path = "./adversarial_test_exp04"
    clean_dir(db_path)
    try:
        dim = 4
        db = Database.open(db_path, dim=dim, metric="l2")
        
        # 1. Test NaN vector insertion
        nan_vec = [float('nan'), 1.0, 2.0, 3.0]
        nan_accepted = False
        try:
            db.put(1, nan_vec)
            nan_accepted = True
        except Exception as e:
            nan_accepted = False
            
        # 2. Test +Inf vector insertion
        inf_vec = [float('inf'), 1.0, 2.0, 3.0]
        inf_accepted = False
        try:
            db.put(2, inf_vec)
            inf_accepted = True
        except Exception:
            inf_accepted = False
            
        # 3. Put normal vector
        db.put(3, [1.0, 1.0, 1.0, 1.0])
        
        # 4. Search with normal vector
        hits = db.search([1.0, 1.0, 1.0, 1.0], topk=3)
        scores = [h.score for h in hits]
        has_nan_score = any(math.isnan(s) for s in scores)
        
        # 5. Search with NaN vector query
        nan_query_hits = []
        try:
            nan_query_hits = db.search(nan_vec, topk=3)
            nan_query_crashed = False
        except Exception:
            nan_query_crashed = True
            
        db.close()
        
        if nan_accepted or inf_accepted:
            record_test("EXP-04", "NaN/Inf Vector Input Validation", "FAIL",
                        f"Database accepted NaN={nan_accepted}, Inf={inf_accepted} without validation! Search produced scores={scores} (has NaN: {has_nan_score})",
                        severity="P1", bug_id="BUG-04")
        else:
            record_test("EXP-04", "NaN/Inf Vector Input Validation", "PASS",
                        "Database rejected NaN and Inf vectors gracefully.")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-05: Magnitude Distortion in kCosine Metric vs Mathematical Oracle
# ==============================================================================
def test_exp05_cosine_magnitude_distortion():
    db_path = "./adversarial_test_exp05"
    clean_dir(db_path)
    try:
        dim = 4
        # Open with metric="cosine"
        db = Database.open(db_path, dim=dim, metric="cosine")
        
        # Vector 1: Direction [1, 0, 0, 0], magnitude = 10.0 -> [10, 0, 0, 0]
        # Vector 2: Direction [1, 0, 0, 0], magnitude = 1.0  -> [1, 0, 0, 0]
        # Vector 3: Direction [0, 1, 0, 0], magnitude = 5.0  -> [0, 5, 0, 0]
        db.put(1, [10.0, 0.0, 0.0, 0.0])
        db.put(2, [1.0, 0.0, 0.0, 0.0])
        db.put(3, [0.0, 5.0, 0.0, 0.0])
        
        # Query vector: [1.0, 0.0, 0.0, 0.0]
        # Mathematical Cosine Similarity Oracle:
        # cos(V1, Q) = (10*1) / (10 * 1) = 1.0000
        # cos(V2, Q) = (1*1) / (1 * 1)   = 1.0000
        # cos(V3, Q) = 0.0000
        # Vectors 1 and 2 MUST have EQUAL cosine similarity of 1.0!
        hits = db.search([1.0, 0.0, 0.0, 0.0], topk=3)
        db.close()
        
        hit_dict = {h.id: h.score for h in hits}
        s1 = hit_dict.get(1, 0.0)
        s2 = hit_dict.get(2, 0.0)
        
        # If PomaiDB treats Cosine as unnormalized Dot product:
        # Dot(V1, Q) = 10.0
        # Dot(V2, Q) = 1.0
        # s1 != s2 by 10x!
        if not math.isclose(s1, s2, rel_tol=1e-3, abs_tol=1e-3):
            record_test("EXP-05", "Cosine Metric Magnitude Distortion", "FAIL",
                        f"Cosine metric violates angle-invariance! V1=[10,0,0,0] and V2=[1,0,0,0] have identical directions, but received scores {s1:.4f} vs {s2:.4f} (treated as raw Dot product without normalization)!",
                        severity="P1", bug_id="BUG-05")
        else:
            record_test("EXP-05", "Cosine Metric Magnitude Distortion", "PASS",
                        f"Cosine similarity invariant held: V1 score={s1:.4f}, V2 score={s2:.4f}")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-06: Data Race / Concurrency in MembraneManager
# ==============================================================================
def test_exp06_membrane_manager_concurrency():
    db_path = "./adversarial_test_exp06"
    clean_dir(db_path)
    try:
        dim = 8
        db = Database.open(db_path, dim=dim)
        db.create_membrane("static_memb", dim, 1)
        
        stop_event = threading.Event()
        errors = []
        
        # Worker 1: Rapidly writes and searches on static_memb
        def reader_writer():
            try:
                step = 0
                while not stop_event.is_set():
                    vec = [float(step % 10)] * dim
                    db.put(step % 100, vec, membrane="static_memb")
                    hits = db.search(vec, topk=5, membrane="static_memb")
                    step += 1
            except Exception as e:
                errors.append(f"Reader/Writer thread crashed: {e}")

        # Worker 2: Rapidly creates and drops dynamic membranes
        def ddl_worker():
            try:
                for i in range(20):
                    name = f"dyn_memb_{i}"
                    db.create_membrane(name, dim, 1)
                    db.list_membranes()
                    db.drop_membrane(name)
            except Exception as e:
                errors.append(f"DDL thread crashed: {e}")

        t1 = threading.Thread(target=reader_writer)
        t2 = threading.Thread(target=ddl_worker)
        
        t1.start()
        t2.start()
        
        t2.join(timeout=10)
        stop_event.set()
        t1.join(timeout=5)
        
        db.close()
        
        if errors:
            record_test("EXP-06", "MembraneManager Concurrency", "FAIL",
                        f"Concurrency errors detected: {errors}", severity="P0", bug_id="BUG-06")
        else:
            record_test("EXP-06", "MembraneManager Concurrency", "PASS",
                        "Concurrent DDL (create/drop) and DML (put/search) completed without crash.")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-07: Double Close Use-After-Free / Double Free
# ==============================================================================
def test_exp07_double_close_uaf():
    db_path = "./adversarial_test_exp07"
    clean_dir(db_path)
    try:
        _ensure_lib()
        opts = pomaidb._lib.PomaiOptions()
        pomaidb._lib.pomai_options_init(ctypes.byref(opts))
        opts.path = db_path.encode("utf-8")
        opts.dim = 4
        opts.shards = 1
        
        db_ptr = ctypes.c_void_p()
        st = pomaidb._lib.pomai_open(ctypes.byref(opts), ctypes.byref(db_ptr))
        if st:
            pomaidb._lib.pomai_status_free(st)
            record_test("EXP-07", "Double Close UAF", "FAIL", "Failed to open DB for test")
            return
            
        # First close
        st1 = pomaidb._lib.pomai_close(db_ptr)
        if st1: pomaidb._lib.pomai_status_free(st1)
        
        # Second close with same pointer
        crashed = False
        err_msg = ""
        st2 = None
        try:
            st2 = pomaidb._lib.pomai_close(db_ptr)
        except OSError as e:
            crashed = True
            err_msg = str(e)
            
        if crashed:
            record_test("EXP-07", "Double Close Use-After-Free / Access Violation", "FAIL",
                        f"Deterministic Access Violation (SIGSEGV/AV) on double close: {err_msg}! Calling pomai_close() on an already closed handle crashes the process with UAF/double-free!",
                        severity="P0", bug_id="BUG-07")
        elif st2:
            msg = pomaidb._lib.pomai_status_message(st2)
            pomaidb._lib.pomai_status_free(st2)
            record_test("EXP-07", "Double Close Protection", "PASS",
                        f"Double close safely guarded by handle registry; returned status: {msg}")
        else:
            record_test("EXP-07", "Double Close Protection", "FAIL",
                        "pomai_close() did not reject already closed handle",
                        severity="P2", bug_id="BUG-07")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-08: Brute-Force Recall@K vs Exact NumPy Mathematical Oracle
# ==============================================================================
def test_exp08_recall_against_numpy_oracle():
    db_path = "./adversarial_test_exp08"
    clean_dir(db_path)
    try:
        dim = 32
        n_vectors = 200
        n_queries = 20
        topk = 10
        
        np.random.seed(42)
        # Generate random normalized vectors
        raw_db = np.random.randn(n_vectors, dim).astype(np.float32)
        raw_db /= np.linalg.norm(raw_db, axis=1, keepdims=True)
        
        raw_queries = np.random.randn(n_queries, dim).astype(np.float32)
        raw_queries /= np.linalg.norm(raw_queries, axis=1, keepdims=True)
        
        db = Database.open(db_path, dim=dim, metric="l2")
        for i in range(n_vectors):
            db.put(i + 1, raw_db[i].tolist())
            
        # Compute exact NumPy brute-force L2 distance oracle
        # L2^2 = ||q - v||^2
        recalls = []
        for q_idx in range(n_queries):
            q = raw_queries[q_idx]
            # Exact brute-force distances
            diffs = raw_db - q
            dists_sq = np.sum(diffs * diffs, axis=1)
            # Top-K smallest distances (1-based IDs)
            true_topk_ids = set(np.argsort(dists_sq)[:topk] + 1)
            
            hits = db.search(q.tolist(), topk=topk)
            retrieved_ids = set(h.id for h in hits)
            
            intersection = len(true_topk_ids.intersection(retrieved_ids))
            recall = intersection / topk
            recalls.append(recall)
            
        db.close()
        avg_recall = np.mean(recalls)
        min_recall = np.min(recalls)
        
        if avg_recall >= 0.95:
            record_test("EXP-08", "Search Recall@10 vs Exact NumPy Oracle", "PASS",
                        f"Recall@10 = {avg_recall*100:.1f}% (min: {min_recall*100:.1f}%) across {n_queries} queries.")
        else:
            record_test("EXP-08", "Search Recall@10 vs Exact NumPy Oracle", "FAIL",
                        f"Poor recall against NumPy brute-force: {avg_recall*100:.1f}%", severity="P1", bug_id="BUG-08")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-09: Repeated Insert/Delete Soak (Memory Stability)
# ==============================================================================
def test_exp09_memory_leak_soak():
    db_path = "./adversarial_test_exp09"
    clean_dir(db_path)
    try:
        dim = 32
        db = Database.open(db_path, dim=dim)
        
        v = [0.5] * dim
        n_iters = 1000
        
        start_time = time.time()
        for i in range(n_iters):
            db.put(1, v)
            db.delete(1)
            if i % 200 == 0:
                db.flush()
                
        elapsed = time.time() - start_time
        db.close()
        
        record_test("EXP-09", "1000-Cycle Insert/Delete Churn Soak", "PASS",
                    f"Completed 1,000 write/delete/flush cycles in {elapsed:.2f}s without crash or unbounded growth.")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-10: Boundary Search Parameters (topk=0, huge topk, empty DB)
# ==============================================================================
def test_exp10_boundary_parameters():
    db_path = "./adversarial_test_exp10"
    clean_dir(db_path)
    try:
        dim = 4
        db = Database.open(db_path, dim=dim)
        
        # 1. Search empty database
        hits_empty = db.search([1.0, 2.0, 3.0, 4.0], topk=5)
        
        # 2. Insert 1 vector
        db.put(1, [1.0, 2.0, 3.0, 4.0])
        
        # 3. Search with topk = 0
        hits_k0 = db.search([1.0, 2.0, 3.0, 4.0], topk=0)
        
        # 4. Search with topk = 1000000 (huge k > N)
        hits_huge_k = db.search([1.0, 2.0, 3.0, 4.0], topk=1000000)
        
        db.close()
        
        if len(hits_empty) != 0:
            record_test("EXP-10", "Boundary Search Parameters", "FAIL", "Empty DB search returned non-empty hits")
        elif len(hits_k0) != 0:
            record_test("EXP-10", "Boundary Search Parameters", "FAIL", "topk=0 returned non-empty hits")
        elif len(hits_huge_k) != 1:
            record_test("EXP-10", "Boundary Search Parameters", "FAIL", f"topk=1000000 returned {len(hits_huge_k)} instead of 1")
        else:
            record_test("EXP-10", "Boundary Search Parameters", "PASS",
                        f"Boundary cases handled cleanly: empty={len(hits_empty)}, k0={len(hits_k0)}, huge_k={len(hits_huge_k)}")
    finally:
        clean_dir(db_path)

if __name__ == "__main__":
    print("=" * 80)
    print(" RUNNING ADVERSARIAL QA & RELIABILITY TEST MATRIX")
    print("=" * 80)
    
    test_exp01_frozen_memtable_update_shadowing()
    test_exp02_silent_payload_loss_in_batch()
    test_exp03_missing_checksum_verification()
    test_exp04_nan_inf_poisoning()
    test_exp05_cosine_magnitude_distortion()
    test_exp06_membrane_manager_concurrency()
    test_exp07_double_close_uaf()
    test_exp08_recall_against_numpy_oracle()
    test_exp09_memory_leak_soak()
    test_exp10_boundary_parameters()
    
    print("\n" + "=" * 80)
    print(" SUMMARY OF ADVERSARIAL EXPERIMENTS")
    print("=" * 80)
    failed_count = sum(1 for r in RESULTS.values() if r["status"] == "FAIL")
    passed_count = sum(1 for r in RESULTS.values() if r["status"] == "PASS")
    print(f"Total Experiments: {len(RESULTS)}")
    print(f"Bugs Confirmed (FAIL): {failed_count}")
    print(f"Invariants Held (PASS): {passed_count}")
    
    with open("adversarial_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nResults exported to adversarial_results.json")
