# PomaiDB Rust SDK (planned GA)

This SDK wraps the C ABI with safe Rust ergonomics.

## Scope
- Open/close database
- Put/search APIs
- Gateway helpers (idempotent ingest + retry)
- Typed error model and result enums

## Conformance
Rust SDK must pass shared golden fixture tests used by Python/Go/Node bindings.
