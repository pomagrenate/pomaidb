# PomaiDB Python Example

This example demonstrates the complete end-to-end feature pipeline for PomaiDB in Python.

## Prerequisites

```bash
pip install pomaidb
```

*(Or when developing in this repository, use the local SDK)*:
```bash
export PYTHONPATH="../../bindings/python"
```

## Running the Example

```bash
python main.py
```

## Features Demonstrated

1. **Engine Initialization**: Configured with SQ8 scalar quantization, Cosine distance metric, memory budget limits, and background flush thresholds.
2. **Multi-Membrane Architecture**: Creating isolated membranes (`realtime_events`, `archival_docs`), listing membranes, and dropping membranes.
3. **Vector CRUD**: Single and batch inserts with high-precision timestamps and arbitrary binary/JSON payloads.
4. **Record Retrieval**: Querying vector dimensions, recovery of binary payloads, and event timestamps.
5. **ANN Vector Search**: Top-K nearest neighbor search with Cosine distance, membrane-scoped search, and batch vector search.
6. **Temporal Queries**: Point-in-time time travel search using `as_of_ts`.
7. **Zero-Copy Memory Access**: High-performance zero-copy session querying.
8. **Engine Maintenance**: Flushes, freezing membranes, and compactions.
9. **Telemetry**: Inspecting engine stats and active membranes via `get_stats()`.
