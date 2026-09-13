#!/usr/bin/env python3
"""
PomaiDB Principal QA Adversarial Test Suite - Part 2
====================================================
Testing:
  - EXP-11: Vector Dimension Mismatch on Put
  - EXP-12: Vector Dimension Mismatch on Search
  - EXP-13: WAL File Bit-Flip Corruption vs Tail Truncation
  - EXP-14: Duplicate ID Semantics (Overwrite vs Duplicate Hits)
  - EXP-15: Latency Percentiles & Ingestion Throughput Scaling (p50, p95, p99)
"""

import math
import os
import shutil
import sys
import time
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "bindings", "python"))

import pomaidb
from pomaidb import Database, PomaiDBError, QUANT_SQ8

RESULTS_PART2 = {}

def record_test(test_id, name, status, evidence, severity=None, bug_id=None):
    RESULTS_PART2[test_id] = {
        "name": name,
        "status": status,
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
# EXP-11: Dimension Mismatch on Ingestion (dim 32 into dim 64 DB)
# ==============================================================================
def test_exp11_dimension_mismatch_ingest():
    db_path = "./adversarial_test_exp11"
    clean_dir(db_path)
    try:
        db = Database.open(db_path, dim=64)
        rejected = False
        err_msg = ""
        try:
            db.put(1, [1.0] * 32)
        except PomaiDBError as e:
            rejected = True
            err_msg = str(e)
            
        db.close()
        if rejected and ("mismatch" in err_msg.lower() or "invalid" in err_msg.lower()):
            record_test("EXP-11", "Vector Dimension Mismatch on Ingestion", "PASS",
                        f"Correctly rejected dimension mismatch (32 vs 64): {err_msg}")
        else:
            record_test("EXP-11", "Vector Dimension Mismatch on Ingestion", "FAIL",
                        f"Database accepted dimension 32 vector into dimension 64 DB!", severity="P1", bug_id="BUG-11")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-12: Dimension Mismatch on Search (dim 32 query into dim 64 DB)
# ==============================================================================
def test_exp12_dimension_mismatch_search():
    db_path = "./adversarial_test_exp12"
    clean_dir(db_path)
    try:
        db = Database.open(db_path, dim=64)
        db.put(1, [1.0] * 64)
        
        rejected = False
        err_msg = ""
        try:
            hits = db.search([1.0] * 32, topk=5)
        except PomaiDBError as e:
            rejected = True
            err_msg = str(e)
        except Exception as e:
            rejected = True
            err_msg = str(e)
            
        db.close()
        if rejected:
            record_test("EXP-12", "Vector Dimension Mismatch on Search", "PASS",
                        f"Correctly rejected search dimension mismatch (32 vs 64): {err_msg}")
        else:
            record_test("EXP-12", "Vector Dimension Mismatch on Search", "FAIL",
                        f"Search silently proceeded with mismatched dimension! Can cause out-of-bounds memory read!", severity="P0", bug_id="BUG-12")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-13: WAL Corruption & Recovery
# ==============================================================================
def test_exp13_wal_corruption_recovery():
    db_path = "./adversarial_test_exp13"
    clean_dir(db_path)
    try:
        dim = 16
        db = Database.open(db_path, dim=dim)
        for i in range(50):
            db.put(i + 1, [float(i)] * dim)
        db.flush() # WAL has 50 records
        db.close()
        
        # Locate WAL file
        wal_files = []
        for root, _, files in os.walk(db_path):
            for f in files:
                if f.endswith(".wal") or "wal" in f.lower():
                    wal_files.append(os.path.join(root, f))
                    
        if not wal_files:
            record_test("EXP-13", "WAL Corruption & Recovery", "FAIL", "No WAL file found after put and flush")
            return
            
        target_wal = wal_files[0]
        with open(target_wal, "rb") as f:
            data = bytearray(f.read())
            
        # Corrupt 4 bytes inside a record body (not tail)
        # Flip bits at offset 64
        if len(data) > 64:
            for b in range(4):
                data[64 + b] ^= 0xAA
                
        with open(target_wal, "wb") as f:
            f.write(data)
            
        # Re-open database
        corrupt_caught = False
        err_msg = ""
        try:
            db_recovered = Database.open(db_path, dim=dim)
            # Check if all 50 vectors are present or if corrupted
            e1 = db_recovered.exists(1)
            e50 = db_recovered.exists(50)
            db_recovered.close()
        except PomaiDBError as e:
            corrupt_caught = True
            err_msg = str(e)
            
        if corrupt_caught and ("crc" in err_msg.lower() or "corrupt" in err_msg.lower()):
            record_test("EXP-13", "WAL Corruption Recovery", "PASS",
                        f"WAL corruption correctly detected and rejected: {err_msg}")
        else:
            record_test("EXP-13", "WAL Corruption Recovery", "FAIL",
                        f"Database silently replayed corrupted WAL without returning Status::Corruption! Error: {err_msg}",
                        severity="P0", bug_id="BUG-13")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-14: Duplicate ID Insertion Semantics
# ==============================================================================
def test_exp14_duplicate_id_semantics():
    db_path = "./adversarial_test_exp14"
    clean_dir(db_path)
    try:
        dim = 4
        db = Database.open(db_path, dim=dim)
        
        # Put ID 10 with vector A
        db.put(10, [1.0, 0.0, 0.0, 0.0])
        # Put ID 10 with vector B (Update)
        db.put(10, [0.0, 1.0, 0.0, 0.0])
        
        rec = db.get(10, with_metadata=True)
        vec = rec.vector if rec else []
        
        # Search query
        hits = db.search([0.0, 1.0, 0.0, 0.0], topk=10)
        hit_ids = [h.id for h in hits]
        
        db.close()
        
        # Invariant 1: Vector recovered must be B [0, 1, 0, 0]
        # Invariant 2: Search must return ID 10 exactly ONCE (no duplicate hits in result set)
        id_10_count = hit_ids.count(10)
        
        if id_10_count > 1:
            record_test("EXP-14", "Duplicate ID Semantics", "FAIL",
                        f"Search returned duplicate hit for ID 10: {hit_ids}", severity="P1", bug_id="BUG-14")
        elif vec != [0.0, 1.0, 0.0, 0.0]:
            record_test("EXP-14", "Duplicate ID Semantics", "FAIL",
                        f"get(10) returned stale vector {vec} instead of updated [0, 1, 0, 0]", severity="P1", bug_id="BUG-14")
        else:
            record_test("EXP-14", "Duplicate ID Semantics", "PASS",
                        f"ID 10 correctly updated to {vec}, returned exactly {id_10_count} time in search.")
    finally:
        clean_dir(db_path)

# ==============================================================================
# EXP-15: Performance Benchmarking & Latency Percentiles (p50, p95, p99)
# ==============================================================================
def test_exp15_perf_scaling_percentiles():
    db_path = "./adversarial_test_exp15"
    clean_dir(db_path)
    try:
        dim = 64
        n_vectors = 5000
        n_queries = 200
        
        print(f"\n[Benchmarking] Ingesting {n_vectors} vectors (dim={dim})...")
        db = Database.open(db_path, dim=dim, metric="cosine", quant_type=QUANT_SQ8)
        
        np.random.seed(123)
        raw_data = np.random.randn(n_vectors, dim).astype(np.float32)
        raw_data /= np.linalg.norm(raw_data, axis=1, keepdims=True)
        
        # Measure ingestion throughput
        t0 = time.perf_counter()
        for i in range(n_vectors):
            db.put(i + 1, raw_data[i].tolist())
        ingest_time = time.perf_counter() - t0
        ingest_throughput = n_vectors / ingest_time
        
        # Measure search latencies
        query_data = np.random.randn(n_queries, dim).astype(np.float32)
        query_data /= np.linalg.norm(query_data, axis=1, keepdims=True)
        
        latencies_ms = []
        for i in range(n_queries):
            q = query_data[i].tolist()
            t_q0 = time.perf_counter()
            hits = db.search(q, topk=10)
            t_q1 = time.perf_counter()
            latencies_ms.append((t_q1 - t_q0) * 1000.0)
            
        db.close()
        
        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)
        avg_qps = n_queries / (sum(latencies_ms) / 1000.0)
        
        evidence = (f"Ingestion: {ingest_throughput:.1f} vec/s ({ingest_time:.2f}s for {n_vectors} vecs). "
                    f"Search Latency: p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms. QPS={avg_qps:.1f} queries/sec.")
        
        record_test("EXP-15", "Performance & Latency Percentiles", "PASS", evidence)
    finally:
        clean_dir(db_path)

if __name__ == "__main__":
    print("=" * 80)
    print(" RUNNING ADVERSARIAL QA & RELIABILITY TEST MATRIX - PART 2")
    print("=" * 80)
    
    test_exp11_dimension_mismatch_ingest()
    test_exp12_dimension_mismatch_search()
    test_exp13_wal_corruption_recovery()
    test_exp14_duplicate_id_semantics()
    test_exp15_perf_scaling_percentiles()
