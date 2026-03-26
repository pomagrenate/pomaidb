# PomaiDB On-Disk Format (v1)

PomaiDB stores data as a directory tree, not a single file.

## Root Layout
- `MANIFEST`: top-level metadata and membrane declarations.
- `gateway/`: edge gateway state (`ingest.seq`, idempotency/audit/sync files).
- `membranes/<name>/`: per-membrane storage.

## Per-Membrane Layout
- `MANIFEST`: membrane-local segment/index references and policy metadata.
- WAL files: append-only durability log.
- Segment files: immutable data/index blocks (`seg_*.dat`).

## Durability Model
1. write accepted
2. WAL append
3. memtable visibility
4. freeze/compact into segments

## Compatibility Rules
- Manifest fields are append-only where possible.
- New fields must have defaults for backward readers.
- Breaking storage format changes require major version migration notes.

## Snapshot Portability
- Export format target: `tar.zst` bundle containing root + membranes + manifests.
- Import must validate checksum and format version before install.
