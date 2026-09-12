# PomaiDB Go Example

This example demonstrates the complete end-to-end feature pipeline for PomaiDB in Go.

## Prerequisites

```bash
go get github.com/pomagrenate/pomaidb/bindings/go
```

## Running the Example

```bash
go run main.go
```

## Features Demonstrated

1. **Engine Initialization**: Configured with SQ8 scalar quantization, Cosine distance metric, memory budget limits.
2. **Multi-Membrane Architecture**: Creating isolated membranes (`realtime_telemetry`, `knowledge_base`), listing membranes, and dropping membranes.
3. **Vector CRUD**: Inserts with high-precision timestamps and arbitrary byte payloads (`[]byte`).
4. **Record Retrieval**: Querying vector dimensions, recovery of binary payloads, and event timestamps.
5. **ANN Vector Search**: Top-K nearest neighbor search with Cosine distance and membrane-scoped search.
6. **Temporal Queries**: Point-in-time time travel search using `AsOfTs`.
7. **Engine Maintenance**: Flushes, freezing membranes, and compactions.
8. **Telemetry**: Inspecting engine stats and active membranes via `GetStats()`.
9. **CGO Safety**: Safe memory management with Go 1.21+ `runtime.Pinner` across native borders.
