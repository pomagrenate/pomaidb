#!/usr/bin/env python3
import time
import os
import shutil
import sys
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "bindings", "python"))

from pomaidb import Database, QUANT_SQ8

def clean_dir(p):
    if os.path.exists(p):
        for _ in range(5):
            try:
                shutil.rmtree(p)
                break
            except Exception:
                time.sleep(0.1)

def run_scale_bench():
    db_path = "./perf_scale_test"
    clean_dir(db_path)
    try:
        dim = 64
        sizes = [1000, 10000, 25000]
        
        print("=" * 80)
        print(" POMAIDB PERFORMANCE SCALING & COMPLEXITY VALIDATION")
        print("=" * 80)
        print(f"{'N':>8} | {'Dim':>4} | {'Ingest (vec/s)':>15} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | {'QPS':>8}")
        print("-" * 80)
        
        for n in sizes:
            db_dir = f"{db_path}_{n}"
            clean_dir(db_dir)
            db = Database.open(db_dir, dim=dim, metric="cosine", quant_type=QUANT_SQ8)
            
            np.random.seed(42)
            data = np.random.randn(n, dim).astype(np.float32)
            data /= np.linalg.norm(data, axis=1, keepdims=True)
            
            # Ingestion
            t0 = time.perf_counter()
            for i in range(n):
                db.put(i + 1, data[i].tolist())
            t1 = time.perf_counter()
            ingest_rate = n / (t1 - t0)
            
            # 100 queries
            queries = np.random.randn(100, dim).astype(np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            
            latencies = []
            for i in range(100):
                q0 = time.perf_counter()
                _ = db.search(queries[i].tolist(), topk=10)
                q1 = time.perf_counter()
                latencies.append((q1 - q0) * 1000.0)
                
            db.close()
            clean_dir(db_dir)
            
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            qps = 100.0 / (sum(latencies) / 1000.0)
            
            print(f"{n:>8} | {dim:>4} | {ingest_rate:>15.1f} | {p50:>10.3f} | {p95:>10.3f} | {p99:>10.3f} | {qps:>8.1f}")
            
    finally:
        clean_dir(db_path)

if __name__ == "__main__":
    run_scale_bench()
