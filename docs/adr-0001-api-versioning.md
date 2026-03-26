# ADR-0001: API Versioning Strategy

## Status
Accepted

## Context
PomaiDB Edge Gateway needs stable external contracts while evolving internal implementation.

## Decision
- `/v1` is strict backward-compatible for request/response fields and semantics.
- Breaking changes are introduced only under `/v2` (or newer major path).
- Error codes are machine-readable and stable once published in `/v1`.

## Consequences
- Contract tests are mandatory before merge for `/v1` handlers.
- Deprecations in `/v1` require at least one release-cycle overlap with migration notes.
