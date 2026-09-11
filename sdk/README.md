# PomaiDB Language SDKs

PomaiDB is an **embedded database** designed to run in-process (like SQLite). It communicates across language boundaries through a high-performance C ABI defined in `include/pomai/c_api.h`.

All foreign language bindings and SDKs reside in this directory:

| Language | Directory | Description | Integration Method |
| :--- | :--- | :--- | :--- |
| **Python** | [`sdk/python/`](./python) | Official Python SDK (`pomaidb`) | Zero-copy ctypes wrapping `libpomai_c` |
| **Go** | [`sdk/go/`](./go) | Go client bindings | CGO binding |
| **Node.js** | [`sdk/node/`](./node) | Node.js bindings | N-API / FFI |
| **Rust** | [`sdk/rust/`](./rust) | Rust safe wrapper | Safe FFI bindings |

## Architecture: Embedded Native C ABI

```text
┌─────────────────────────────────────────────────────────────┐
│                       User Application                      │
│        (Python / Go / Rust / Node.js / C++ / C)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │   Language SDK (e.g. sdk/python)    │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │  PomaiDB C ABI (include/pomai/c_api) │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │         libpomai_c.so / .dll         │
            │   (Built with palloc core allocator)│
            └─────────────────────────────────────┘
```

## Building and Testing the SDKs

### Python
```bash
# Install in editable mode
pip install -e sdk/python/

# Run Python SDK completeness test
PYTHONPATH="sdk/python;build" python tests/python_api_completeness_test.py
```
