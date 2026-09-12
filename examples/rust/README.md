# PomaiDB Rust Example

This example demonstrates the complete end-to-end feature pipeline for PomaiDB in Rust.

## Prerequisites

Add PomaiDB to your `Cargo.toml`:
```toml
[dependencies]
pomaidb = "0.1.0"
```

## Running the Example

```bash
cargo run
```

## Features Demonstrated

1. **Engine Initialization**: Configured with SQ8 scalar quantization, Cosine distance metric, and memory budget limits.
2. **Multi-Membrane Architecture**: Creating isolated membranes (`realtime_telemetry`, `knowledge_base`), listing membranes, and dropping membranes.
3. **Vector CRUD**: Inserts with high-precision timestamps and arbitrary byte payloads.
4. **Record Retrieval**: Querying vector dimensions, recovery of binary payloads, and event timestamps.
5. **ANN Vector Search**: Top-K nearest neighbor search with Cosine distance and membrane-scoped search.
6. **Temporal Queries**: Point-in-time time travel search using `as_of_ts`.
7. **Engine Maintenance**: Flushes, freezing membranes, and compactions.
8. **Telemetry**: Inspecting engine stats and active membranes via `get_stats()`.
9. **RAII Safety**: Clean destruction and thread-safe handle management.
