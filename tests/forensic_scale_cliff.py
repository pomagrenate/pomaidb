"""
Scale Test to find the Scaling Cliff: N = 50K, 100K
"""

import os
import sys
import time
import shutil
import numpy as np

sys.path.insert(0, os.path.abspath("bindings/python"))
from pomaidb import Database, QUANT_SQ8

def clean_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)

def scale_test():
    dim = 64
    n_queries = 50
    topk = 10
    
    for n in [50000, 100000]:
        db_path = f"./forensic_db_scale_{n}"
        clean_dir(db_path)
        
        print(f"\n[Testing Scale] N = {n}, D = {dim}...")
        np.random.seed(n)
        vectors = np.random.randn(n, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        db = Database.open(db_path, dim=dim, metric="cosine", quant_type=QUANT_SQ8)
        
        t0 = time.perf_counter()
        batch_size = 1000
        for b in range(0, n, batch_size):
            for i in range(b, min(n, b + batch_size)):
                db.put(i + 1, vectors[i].tolist())
        ingest_sec = time.perf_counter() - t0
        ingest_qps = n / ingest_sec
        print(f"  Ingestion: {ingest_qps:.0f} vectors/sec ({ingest_sec:.2f}s)")
        
        t_c0 = time.perf_counter()
        db.freeze()
        db.compact()
        compact_sec = time.perf_counter() - t_c0
        print(f"  Compaction: {compact_sec:.2f}s")
        
        # Measure search
        lats = []
        for q in queries:
            t_q0 = time.perf_counter()
            hits = db.search(q.tolist(), topk=topk)
            lats.append((time.perf_counter() - t_q0) * 1000.0)
            
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        p99 = np.percentile(lats, 99)
        qps = n_queries / (sum(lats) / 1000.0)
        print(f"  Search: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms, QPS={qps:.0f}")
        
        db.close()
        clean_dir(db_path)

if __name__ == "__main__":
    scale_test()
