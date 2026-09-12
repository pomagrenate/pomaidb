# PomaiDB JavaScript / Node.js Example

This example demonstrates the complete end-to-end feature pipeline for PomaiDB in JavaScript / Node.js.

## Prerequisites

```bash
npm install pomaidb
```

*(Or when developing in this repository, install the local package)*:
```bash
npm install
```

## Running the Example

```bash
node index.js
```

## Features Demonstrated

1. **Engine Initialization**: Configured with SQ8 scalar quantization, Cosine distance metric, memory budget limits.
2. **Multi-Membrane Architecture**: Creating isolated membranes (`realtime_telemetry`, `knowledge_base`), listing membranes, and dropping membranes.
3. **Vector CRUD**: Inserts with high-precision timestamps and arbitrary binary/JSON `Buffer` payloads.
4. **Record Retrieval**: Querying vector dimensions, recovery of binary payloads, and event timestamps.
5. **ANN Vector Search**: Top-K nearest neighbor search with Cosine distance and membrane-scoped search.
6. **Temporal Queries**: Point-in-time time travel search using `asOfTs`.
7. **Engine Maintenance**: Flushes, freezing membranes, and compactions.
8. **Telemetry**: Inspecting engine stats and active membranes via `getStats()`.
