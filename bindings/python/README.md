# PomaiDB Python Bindings

Official Python bindings for **PomaiDB**, an embedded vector database designed for high-performance Edge AI applications.

## Installation

```bash
pip install pomaidb
```

## Quick Start

```python
import pomaidb

# 1. Open database
db = pomaidb.open_db("test_db", dim=4)

# 2. Put vectors
db.put(1, [1.0, 0.0, 0.0, 0.0])
db.put(2, [0.0, 1.0, 0.0, 0.0], membrane="docs", payload=b"doc_payload", timestamp=1000)

# 3. Search
hits = db.search([1.0, 0.0, 0.0, 0.0], topk=5)
for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}")

# 4. Multi-membrane query
hits_docs = db.search([0.0, 1.0, 0.0, 0.0], topk=5, membrane="docs")

# 5. Flush and close
db.flush()
db.close()
```

## Features
- In-process embedded execution (zero network overhead, zero configuration).
- Pomegranate Engine Architecture: Rind MemTable, Locule immutable containers, and Press compaction.
- Built-in Quantization: FP32, SQ8, FP16, 1-bit binary quantization, PQ8.
- Native multi-membrane tenancy and arbitrary payload buffering.
