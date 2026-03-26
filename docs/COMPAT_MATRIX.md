# PomaiDB Compatibility Matrix

## Operating Systems
- Linux x86_64: supported
- Linux arm64: supported (embedded + server)
- macOS arm64/x86_64: supported (embedded + server)
- Windows x86_64: supported (embedded + server)

## Runtime Modes
- Embedded mode: C/C++ API, C ABI, Python bindings.
- Server mode: `pomaidb_server` with Edge Gateway (`/v1/*`, ingest endpoints).

## GPU/Vulkan
- Vulkan is optional.
- CPU-only systems are supported.
- CI can skip Vulkan-specific tests with `POMAI_SKIP_VULKAN_TESTS=1`.

## CPU Architectures
- x86_64: primary support target.
- arm64: supported; throughput may vary by SIMD and memory profile.

## Feature Notes
- Multi-modal ingest: vector/graph/mesh/audio/timeseries/keyvalue/document/rag.
- Replication: upstream sync queue with checkpointing.
- Lifecycle: TTL/retention policy support by membrane kind.
