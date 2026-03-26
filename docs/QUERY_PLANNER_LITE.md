# Query Planner Lite (Multi-Modal)

Planner-lite provides a deterministic path for operator-visible explain output.

## Default Path
1. vector prefilter
2. graph rerank (optional)
3. document evidence assembly (optional)

## Operator UX
- `pomaictl explain` prints selected path.
- `pomaictl query` runs local vector search batch helper.
- `pomaictl replay` replays ndjson sync events into ingest APIs.
- `pomaictl inspect-segment` prints segment files and sizes.
