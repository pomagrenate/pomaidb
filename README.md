<div align="center">

# PomaiDB

<img src="./assets/logo.png" alt="PomaiDB logo" width="200">

### The predictable, edge-native database for multimodal AI memory.

**Embedded vector search · Offline RAG · Multimodal storage · ARM64 · Zero-OOM design**

<p>
  <a href="#-why-pomaidb">Why PomaiDB</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-performance">Performance</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-quick-start">Quick Start</a>
</p>

<p>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=for-the-badge&logo=c%2B%2B" alt="C++20">
  <img src="https://img.shields.io/badge/ARM64-Edge_AI-green.svg?style=for-the-badge" alt="ARM64 Edge AI">
  <img src="https://img.shields.io/badge/RAG-Offline-orange.svg?style=for-the-badge" alt="Offline RAG">
</p>

</div>

---

## 🚀 What is PomaiDB?

**PomaiDB is a lightweight, embeddable vector database and multimodal AI memory engine built for edge devices.**

It brings **vector search, semantic retrieval, offline RAG, graph relationships, mesh data, time-series data, key-value storage, and multimodal memory** into a single native database designed to run directly inside your application.

Unlike cloud-first and distributed vector databases, PomaiDB is designed around a different assumption:

> **One process. One logical database. One predictable execution path.**

That constraint is intentional.

PomaiDB prioritizes:

* Predictable latency
* Low memory usage
* Sequential storage I/O
* Flash-storage longevity
* Deterministic execution
* Offline operation
* Native ARM64 deployment
* Embeddability
* Zero-OOM-oriented resource management

It is designed for **Raspberry Pi, Orange Pi, NVIDIA Jetson, NAS devices, gateways, cameras, custom operating systems, and other resource-constrained edge hardware**.

---

# 🎯 Why PomaiDB?

Traditional vector databases are often designed around servers, distributed deployments, high concurrency, and large memory footprints.

Edge AI has different requirements.

A camera gateway may have:

```text
Limited RAM
    +
SD / eMMC storage
    +
Intermittent connectivity
    +
Local AI inference
    +
Strict latency requirements
```

Running a cloud-oriented database on that hardware can introduce unnecessary complexity.

PomaiDB takes the opposite approach:

```text
Embedded
   ↓
Single-threaded
   ↓
Predictable
   ↓
Memory bounded
   ↓
Flash-friendly
   ↓
Offline-first
   ↓
Edge-native
```

### The result

A database that can live **inside your AI application instead of beside it**.

---

# ✨ Features

## 🧠 Vector Search

PomaiDB provides native vector ingestion and similarity search with configurable ANN indexes.

Supported capabilities include:

* Approximate nearest-neighbor search
* IVF
* HNSW
* Batch search
* Point queries
* Metadata filtering
* Scalar quantization
* SQ8
* Configurable vector dimensions

This makes PomaiDB suitable for semantic search, embeddings, recommendation systems, visual similarity, and local AI memory.

---

## 🤖 Offline-First RAG

PomaiDB includes an embedded RAG pipeline designed to operate entirely on-device.

```text
Document
   │
   ▼
Chunk
   │
   ▼
Embedding
   │
   ▼
Vector Storage
   │
   ▼
Semantic Search
   │
   ▼
Retrieved Context
   │
   ▼
Local AI Model
```

No cloud API is required.

The RAG pipeline provides:

* Zero-copy chunking
* `EmbeddingProvider`
* Document ingestion
* Context retrieval
* Configurable chunk limits
* Bounded batch sizes
* Local-first execution
* C++ and Python APIs

The architecture is also designed so a small local embedding model such as a GGML/`llama.cpp`-based model can be integrated without changing the RAG pipeline itself.

---

# 🌐 Edge-Native Connectivity

PomaiDB can expose lightweight embedded endpoints for applications that need direct edge connectivity.

Available functionality includes:

```text
/health
/metrics
/ingest/meta/...
/ingest/vector/...
```

The edge connectivity layer supports:

* HTTP ingestion
* Lightweight MQTT/WebSocket-style ingestion
* Bearer-token authentication
* Rate limiting
* JSON error responses
* Idempotency keys
* ACK / ERR responses

This makes PomaiDB suitable for **edge gateways, IoT devices, sensors, cameras, and local AI systems**.

---

# 🧩 Multimodal Memory

PomaiDB does not restrict memory to vectors.

Its membrane architecture supports multiple data types:

```text
kVector
kRag
kGraph
kText
kTimeSeries
kKeyValue
kMeta
kSketch
kBlob
kSpatial
kMesh
kSparse
kBitset
```

Each membrane can have its own dimensions, indexes, and storage behavior.

This allows a single embedded database to represent relationships between different types of AI memory.

---

# 🔗 Object Linking

PomaiDB supports shared global identifiers across different memory representations.

For example:

```text
Vector Hit
    │
    ├── Graph Vertex
    │
    └── Mesh Object
```

The `ObjectLinker` allows vector search results to expand into related graph and mesh objects.

This is particularly useful for multimodal AI systems where a single entity may have:

* Text embeddings
* Visual embeddings
* Graph relationships
* 3D geometry
* Metadata

---

# 🕸️ Hybrid & Multimodal Search

`QueryOrchestrator` can combine multiple retrieval strategies:

```text
             Query
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
    Vector   Lexical   Graph
       │       │        │
       └───────┼────────┘
               ▼
       Query Orchestrator
               │
               ▼
        Ranked Results
```

Execution can use:

* Vector search
* Lexical search
* Graph traversal
* Metadata partition hints
* Bounded query frontiers
* Heuristic execution ordering

This enables hybrid retrieval without requiring multiple external databases.

---

# 🗃️ More Than a Vector Database

PomaiDB is designed as a **multimodal embedded memory layer**, not just a vector index.

Supported capabilities include:

| Capability  | Purpose                              |
| ----------- | ------------------------------------ |
| Vector      | Embeddings and similarity search     |
| RAG         | Local retrieval-augmented generation |
| Graph       | Relationships and traversal          |
| Text        | Text storage and lexical retrieval   |
| Time Series | Temporal data                        |
| Key-Value   | Fast structured lookup               |
| Meta        | Metadata                             |
| Sketch      | Compact analytical structures        |
| Blob        | Binary objects                       |
| Spatial     | Spatial information                  |
| Mesh        | 3D mesh data                         |
| Sparse      | Sparse representations               |
| Bitset      | Compact boolean/set operations       |

---

# 🦣 Edge-Friendly by Design

## SD-Card Savior

Flash storage behaves differently from enterprise SSDs.

SD cards and eMMC devices have limited write endurance, and random writes can increase write amplification and wear.

PomaiDB therefore uses an **append-only, log-structured storage model**.

```text
New Data
   │
   ▼
Append to Log
   │
   ▼
Sequential Storage
```

Deletes and updates are represented through tombstones rather than destructive in-place rewrites.

The result is an I/O model designed specifically with **consumer flash and edge storage longevity** in mind.

---

# 🧵 Single-Threaded by Choice

PomaiDB deliberately avoids unnecessary concurrency complexity.

The database uses a **single-threaded event loop**:

```text
Ingest
  ↓
Search
  ↓
Freeze
  ↓
Flush
  ↓
Maintenance
```

Operations execute deterministically.

That means:

* No mutex-heavy hot path
* No lock-free queue complexity
* No database-level race conditions
* No deadlock management
* Predictable operation ordering
* Better CPU cache locality

The design is conceptually similar to the single-threaded execution model used by systems such as Redis and Node.js.

---

# 🛑 Zero-OOM Philosophy

Edge devices cannot afford uncontrolled memory growth.

PomaiDB therefore treats memory as a first-class resource.

The runtime supports:

* Bounded memtables
* Automatic backpressure
* Auto-freeze thresholds
* Configurable query frontiers
* Document size limits
* Chunk size limits
* Batch limits
* Optional hard memory caps
* `palloc` integration
* Bounded KV, blob, sketch, and text storage

The Edge RAG pipeline is designed to operate within constrained environments, including configurations targeting approximately **64–128 MB RAM**.

---

# 🔐 Edge Security

PomaiDB includes security hardening designed for embedded deployments.

Current capabilities include:

* AES-256-GCM encryption at rest
* WAL encryption
* Key manager primitives
* Key wipe support
* Anomaly-triggered key wipe hooks
* Authentication for edge ingestion
* Rate limiting
* Idempotency keys

Security is treated as part of the database runtime rather than an external infrastructure requirement.

---

# 🏗️ Architecture

PomaiDB follows a deliberately small architecture.

```text
                    Application
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           C++           C          Python
             │           │           │
             └───────────┼───────────┘
                         ▼
                    PomaiDB API
                         │
                         ▼
                  ┌──────────────┐
                  │   DB / Core  │
                  └──────┬───────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        Membrane      Query       Storage
        Manager       Planner      Engine
              │          │          │
              ▼          ▼          ▼
        Multimodal   Orchestrator   WAL
        Membranes        │          │
                         ▼          ▼
                    Hybrid Search  Segments
```

Core components include:

* `DbImpl`
* `MembraneManager`
* `QueryPlanner`
* `QueryOrchestrator`
* Typed membrane APIs
* Log-structured storage
* VFS abstraction
* Optional `palloc`
* RAG pipeline

---

# 💾 Storage Architecture

PomaiDB uses a log-structured, append-only storage model.

```text
Application
     │
     ▼
  Memtable
     │
     │ bounded
     ▼
   Freeze
     │
     ▼
   Flush
     │
     ▼
  WAL / Segments
```

The storage layer provides:

* Sequential writes
* Tombstone-based deletion
* Write-behind
* Optional explicit `Flush()`
* Zero-copy reads
* Memory mapping where supported
* Buffered I/O fallback
* VFS abstraction

The VFS layer abstracts operating-system-specific file operations through interfaces such as:

```text
Env
SequentialFile
RandomAccessFile
WritableFile
FileMapping
```

This keeps the core storage engine independent from direct POSIX system calls.

---

# 📐 Membranes

A **membrane** is PomaiDB's logical abstraction for organizing different classes of data.

Membranes provide isolation for:

* Dimensions
* Indexes
* Storage
* Query behavior
* Data types

Example:

```text
Database
│
├── images
│   └── kVector
│
├── documents
│   └── kRag
│
├── entities
│   └── kGraph
│
└── meshes
    └── kMesh
```

This makes it possible to build multimodal AI memory systems without operating several independent databases.

---

# 🧠 Hardware Support

PomaiDB targets edge and embedded environments first.

### ARM64

Designed for platforms such as:

* Raspberry Pi
* Orange Pi
* NVIDIA Jetson
* ARM64 gateways
* Embedded Linux devices

### x86_64

Also suitable for:

* Developer workstations
* Edge servers
* NAS systems
* Small servers
* Local AI machines

The single-threaded architecture avoids NUMA and CPU-pinning complexity.

---

# 📊 Performance

PomaiDB is optimized for **predictable latency and constrained hardware**, not maximum distributed throughput.

The latest benchmark suite was executed through:

```bash
./tools/run_benchmarks_one_by_one.sh
```

with the full suite completing successfully.

### Benchmark Hardware

```text
Device:  HP ProBook 450 G5
CPU:     Intel Core i7-8550U @ 1.80GHz
Cores:   8
RAM:     16 GB
Storage: SATA SSD
```

### Latest Results

| Benchmark                 | Workload                            |                  Result |
| ------------------------- | ----------------------------------- | ----------------------: |
| **Comprehensive Search**  | 10K vectors / 1K queries / top-k=10 |       **18.55 ms mean** |
| **Comprehensive Search**  | P99 latency                         |            **28.52 ms** |
| **Comprehensive Search**  | Throughput                          |           **53.89 QPS** |
| **Comprehensive Search**  | Recall@10                           |                **100%** |
| **Ingestion**             | 10K × 128-dim vectors               |  **31,004 vectors/sec** |
| **RAG Lexical**           | Chunked retrieval                   |            **0.068 ms** |
| **RAG Hybrid**            | Lexical + vector                    |            **0.064 ms** |
| **CI Performance Gate**   | 2K vectors / 300 queries            | **56,847.3 QPS ingest** |
| **Low-Memory Edge Churn** | 5 constrained cycles                |   **51.2 MiB peak RSS** |
| **Mesh Auto-LOD**         | 4,096 triangles / 5K operations     |          **276.005 ms** |

> **Note:** Benchmark results depend on CPU, storage medium, filesystem, `fsync` policy, workload, and configured memory limits. Do not treat these numbers as universal performance guarantees.

---

# ⚙️ Installation

## Requirements

PomaiDB requires:

* C++20 compiler
* CMake 3.20+
* Git
* Vulkan headers for mesh-related functionality

The Khronos Vulkan headers are expected at:

```text
third_party/vulkan/include/
```

CMake reports the detected Vulkan header location during configuration.

---

## Clone

```bash
git clone --recursive https://github.com/pomagrenate/pomaidb
cd pomaidb
```

## Build

```bash
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

# 🧪 Run Tests

Build the full test suite:

```bash
cmake -S . \
  -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPOMAI_BUILD_TESTS=ON

cmake --build build -j$(nproc)
```

Run tests:

```bash
ctest \
  --test-dir build \
  --output-on-failure
```

---

# 📦 Smaller Clone for Embedded / CI

For a smaller checkout:

```bash
git clone \
  --depth 1 \
  --recursive \
  https://github.com/pomagrenate/pomaidb

cd pomaidb

./tools/slim_palloc_submodule.sh
```

Then build using the normal CMake workflow.

---

# 🏁 Quick Start

## C++20

```cpp
#include "pomai/pomai.h"

#include <cstdio>
#include <memory>
#include <vector>

int main() {
    pomai::DBOptions opt;
    opt.path = "/data/vectors";
    opt.dim = 384;
    opt.shard_count = 1;

    std::unique_ptr<pomai::DB> db;

    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) return 1;

    std::vector<float> vec(opt.dim, 0.1f);

    db->Put(1, vec);
    db->Flush();
    db->Freeze("__default__");

    pomai::SearchResult result;
    db->Search(vec, 5, &result);

    for (const auto& hit : result.hits) {
        std::printf(
            "id=%llu score=%.4f\n",
            (unsigned long long)hit.id,
            hit.score
        );
    }

    db->Close();

    return 0;
}
```

---

# 🤖 Quick Start: Edge RAG

```cpp
#include "pomai/pomai.h"
#include "pomai/rag/embedding_provider.h"
#include "pomai/rag/pipeline.h"

int main() {
    pomai::DBOptions opt;
    opt.path = "/tmp/rag_db";
    opt.dim = 4;

    std::unique_ptr<pomai::DB> db;

    pomai::DB::Open(opt, &db);

    pomai::MembraneSpec rag;
    rag.name = "rag";
    rag.kind = pomai::MembraneKind::kRag;

    db->CreateMembrane(rag);

    pomai::MockEmbeddingProvider provider(4);

    pomai::RagPipeline pipeline(
        db.get(),
        "rag",
        4,
        &provider
    );

    pipeline.IngestDocument(
        1,
        "Your document text here."
    );

    std::string context;

    pipeline.RetrieveContext(
        "your query",
        5,
        &context
    );

    db->Close();

    return 0;
}
```

---

# 🐍 Quick Start: Python

```python
import pomaidb

db = pomaidb.open_db(
    "/tmp/rag_db",
    dim=4,
)

pomaidb.create_rag_membrane(
    db,
    "rag",
    dim=4,
)

pomaidb.ingest_document(
    db,
    "rag",
    doc_id=1,
    text="Your document text here.",
)

context = pomaidb.retrieve_context(
    db,
    "rag",
    "your query",
    top_k=5,
)

pomaidb.close(db)
```

---

# 💡 Use Cases

PomaiDB is designed for applications where **AI memory needs to live close to the data**.

## 📸 Edge Computer Vision

Store embeddings from:

* Camera frames
* Object crops
* Images
* Video segments

Then perform similarity search directly on the device.

---

## 🧠 Local AI Memory

Build local AI systems that need persistent memory without sending every query to the cloud.

```text
User / Sensor
      ↓
Local AI
      ↓
PomaiDB
      ↓
Vector / RAG / Graph Memory
      ↓
Local Retrieval
```

---

## 📚 Offline RAG

Run retrieval-augmented generation entirely on-device.

Ideal for:

* Offline assistants
* Private document search
* Industrial systems
* Field devices
* Local knowledge bases
* Air-gapped environments

---

## 🔍 Offline Semantic Search

Index documents and media locally and perform semantic retrieval without depending on external infrastructure.

---

## 📡 IoT & Edge Gateways

Store and retrieve sensor data directly at the edge.

PomaiDB is particularly suitable when:

```text
Network unavailable
       OR
Cloud unavailable
       OR
Data must stay local
```

---

## 🖥️ NAS & Home AI

Use PomaiDB as a local semantic memory layer for:

* NAS systems
* Personal AI
* Home automation
* Local search
* Private knowledge bases

---

## 🔧 Custom Operating Systems

The VFS architecture allows the storage environment to be replaced with:

* POSIX
* In-memory backends
* Custom filesystem implementations
* Non-POSIX environments

---

# 📖 Documentation

More detailed documentation is available in the repository:

| Document                                            | Description           |
| --------------------------------------------------- | --------------------- |
| [`EDGE_RELEASE.md`](docs/EDGE_RELEASE.md)           | Edge release criteria |
| [`EDGE_DEPLOYMENT.md`](docs/EDGE_DEPLOYMENT.md)     | Edge deployment       |
| [`FAILURE_SEMANTICS.md`](docs/FAILURE_SEMANTICS.md) | Failure behavior      |
| [`PYTHON_API.md`](docs/PYTHON_API.md)               | Python API            |
| [`VERSIONING.md`](docs/VERSIONING.md)               | ABI and versioning    |

Examples for C++, C ABI, Python, Go, JavaScript/TypeScript, and RAG are available in [`examples/README.md`](examples/README.md).

---

# 🧭 When Should You Use PomaiDB?

Choose PomaiDB when you need:

```text
Embedded database
        +
Vector search
        +
Local AI / RAG
        +
Low memory
        +
ARM64
        +
Offline operation
        +
Flash-friendly storage
```

PomaiDB is especially compelling when the database needs to run **inside the same process as your application**.

---

# 🚫 What PomaiDB Is Not

PomaiDB is intentionally not designed to replace every database.

It is **not** primarily optimized for:

* Distributed clusters
* Multi-node consensus
* Massive concurrent workloads
* Cloud-scale horizontal sharding
* Maximum throughput at any cost

If your system needs a distributed vector database serving thousands of concurrent clients across many nodes, a server-oriented database may be a better choice.

PomaiDB exists for a different problem:

> **Reliable AI memory on the edge.**

---

# 🔥 The PomaiDB Philosophy

Most databases ask:

> How fast can we scale?

PomaiDB asks:

> **How reliably can we run on the device that actually has to do the work?**

That changes the engineering priorities.

```text
Cloud Database
    ↓
Scale
Concurrency
Throughput
Distributed infrastructure
```

versus:

```text
PomaiDB
    ↓
Predictability
Memory bounds
Storage endurance
Offline operation
Embeddability
Edge reliability
```

For edge AI, those constraints are not limitations.

**They are the product.**

---

# 👥 Contributors

<div align="center">

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/pomagrenate">
        <img src="https://github.com/pomagrenate.png" width="80" alt="pomagrenate">
        <br>
        <sub><b>pomagrenate</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/quanvanskipli">
        <img src="https://github.com/quanvanskipli.png" width="80" alt="quanvanskipli">
        <br>
        <sub><b>quanvanskipli</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/claude">
        <img src="https://github.com/claude.png" width="80" alt="claude">
        <br>
        <sub><b>claude</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Roto0flame">
        <img src="https://github.com/Roto0flame.png" width="80" alt="Roto0flame">
        <br>
        <sub><b>Roto0flame</b></sub>
      </a>
    </td>
  </tr>
</table>

</div>

---

# ⭐ Star History

<div align="center">

<a href="https://star-history.com/#pomagrenate/pomaidb&Date">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://api.star-history.com/svg?repos=pomagrenate/pomaidb&type=Date&theme=dark"
    />
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://api.star-history.com/svg?repos=pomagrenate/pomaidb&type=Date"
    />
    <img
      src="https://api.star-history.com/svg?repos=pomagrenate/pomaidb&type=Date"
      alt="PomaiDB GitHub Star History"
      width="100%"
    />
  </picture>
</a>

</div>

---

# 🔎 Keywords

**PomaiDB**, **embedded vector database**, **C++20 vector database**, **edge AI database**, **edge-native database**, **multimodal AI memory**, **offline RAG database**, **embedded RAG**, **vector search**, **semantic search**, **similarity search**, **ARM64 database**, **Raspberry Pi vector database**, **Jetson AI database**, **Orange Pi database**, **local AI memory**, **offline semantic search**, **single-threaded database**, **append-only database**, **log-structured storage**, **zero-copy database**, **mmap database**, **low-memory database**, **zero-OOM database**, **SD-card optimized database**, **eMMC-friendly database**, **IoT database**, **embedded AI**, **edge RAG**, **VFS**, **virtual file system**, **HNSW**, **IVF**, **SQ8**, **C++ vector database**, **Python vector database**.

---

# 📜 License

PomaiDB is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

<div align="center">

## Build AI Memory for the Edge.

**No cloud required.
No distributed cluster required.
No massive infrastructure required.**

### PomaiDB

**Predictable. Embedded. Multimodal. Edge-native.**

⭐ Star the repository if PomaiDB is useful to you.

</div>
