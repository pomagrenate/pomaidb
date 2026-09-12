# PomaiDB Multi-Language Pipeline Examples

This directory contains comprehensive, end-to-end pipeline examples demonstrating 100% of PomaiDB's capabilities across all 4 officially supported language bindings:

| Language | Directory | Key Technology |
| :--- | :--- | :--- |
| **Python** | [`examples/python/`](./python/) | `ctypes` C-API + Pythonic `Database` API |
| **JavaScript / Node.js** | [`examples/javascript/`](./javascript/) | ES Modules + `koffi` C-FFI |
| **Rust** | [`examples/rust/`](./rust/) | Idiomatic Rust RAII + `sys` C ABI |
| **Go** | [`examples/go/`](./go/) | CGO + `runtime.Pinner` memory safety |

---

## Capabilities Demonstrated in Every Example

Every example script executes a 10-step full pipeline covering the complete PomaiDB feature set:

1. **Engine Initialization & Options**:
   - Scalar Quantization (`SQ8` / `FP16` / `PQ8` / `Bit` / `None`)
   - Distance Metrics (`Cosine`, `InnerProduct`, `L2`)
   - Memory budgets & background MemTable thresholds
2. **Multi-Membrane Architecture**:
   - Creating custom isolated membranes (e.g. `realtime_telemetry`, `knowledge_base`)
   - Querying active membranes (`list_membranes`)
   - Dropping membranes
3. **Vector CRUD & Payloads**:
   - Inserting vectors into default and custom membranes
   - Storing high-precision 64-bit event timestamps
   - Storing opaque binary payloads (JSON metadata, byte buffers)
4. **Existence & Record Retrieval**:
   - Fast index existence checks (`exists`)
   - Full record retrieval recovering vector coordinates, dimensions, payloads, and timestamps
5. **Top-K ANN Vector Search**:
   - High-performance approximate nearest neighbor search
   - Membrane-scoped search targeting isolated partitions
6. **Point-in-Time Temporal Queries**:
   - Time-travel querying vectors using `as_of_ts`
7. **Engine Maintenance Operations**:
   - Explicit WAL / MemTable flushing to disk (`flush`)
   - Freezing active write buffers (`freeze`)
   - Triggering background compactions (`compact`)
8. **Engine Telemetry & Statistics**:
   - Retrieving live JSON engine health, ABI version, and membrane list (`get_stats`)
9. **Vector Deletion**:
   - Deleting entries from default or specific membranes
10. **Clean Shutdown & Resource Management**:
    - Safe database closure, unmapping files, and freeing native allocations

---

## Quick Start

### 1. Python Example
```bash
cd examples/python
python main.py
```

### 2. JavaScript / Node.js Example
```bash
cd examples/javascript
npm install
node index.js
```

### 3. Rust Example
```bash
cd examples/rust
cargo run
```

### 4. Go Example
```bash
cd examples/go
go run main.go
```
