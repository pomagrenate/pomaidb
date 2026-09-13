"""
Concurrency & Contention Forensic Benchmark
Measures multi-threaded query scaling: 1, 2, 4, 8 threads
"""

import os
import sys
import time
import shutil
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.abspath("bindings/python"))
from pomaidb import Database, QUANT_SQ8

def clean_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)

def benchmark_concurrency():
    dim = 64
    n_vectors = 10000
    queries_per_thread = 100
    topk = 10
    db_path = "./forensic_db_concurrency"
    clean_dir(db_path)
    
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    db = Database.open(db_path, dim=dim, metric="cosine", quant_type=QUANT_SQ8)
    for i in range(n_vectors):
        db.put(i + 1, vectors[i].tolist())
    db.freeze()
    db.compact()
    
    print("\n" + "="*80)
    print(" CONCURRENCY & CONTENTION FORENSIC BENCHMARK (1, 2, 4, 8 Threads)")
    print(f" Dataset: N={n_vectors}, D={dim}, Queries/thread={queries_per_thread}")
    print("="*80)
    
    results = []
    
    for num_threads in [1, 2, 4, 8]:
        thread_queries = []
        for t in range(num_threads):
            qs = np.random.randn(queries_per_thread, dim).astype(np.float32)
            qs /= np.linalg.norm(qs, axis=1, keepdims=True)
            thread_queries.append([q.tolist() for q in qs])
            
        def worker(q_list):
            lats = []
            for q in q_list:
                t0 = time.perf_counter()
                hits = db.search(q, topk=topk)
                lats.append((time.perf_counter() - t0) * 1000.0)
            return lats
            
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, thread_queries[t]) for t in range(num_threads)]
            all_lats = []
            for f in futures:
                all_lats.extend(f.result())
        total_time = time.perf_counter() - t0
        total_queries = num_threads * queries_per_thread
        qps = total_queries / total_time
        p50 = np.percentile(all_lats, 50)
        p95 = np.percentile(all_lats, 95)
        p99 = np.percentile(all_lats, 99)
        
        results.append((num_threads, qps, p50, p95, p99))
        print(f"Threads: {num_threads:2d} | QPS: {qps:6.1f} | p50: {p50:5.2f} ms | p95: {p95:5.2f} ms | p99: {p99:5.2f} ms")
        
    db.close()
    clean_dir(db_path)
    return results

if __name__ == "__main__":
    benchmark_concurrency()
