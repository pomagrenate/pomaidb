# Replication Envelope and Replay

PomaiDB sync events are serialized into a signed envelope:

```json
{
  "sig": "<stable_hash(payload + secret)>",
  "payload": {
    "schema_version": 1,
    "seq": 123,
    "type": "vector_put",
    "membrane": "default",
    "id": 42,
    "id2": 0,
    "u32_a": 0,
    "u32_b": 0,
    "aux_k": "",
    "aux_v": "0.1,0.2,0.3"
  }
}
```

## Reliability Semantics
- Retry with exponential backoff.
- Dead-letter after max retry count.
- Persist checkpoint by sequence.

## Observability
- `sync_queue_depth`
- `sync_lag_events`
- `sync_dead_letter_total`
- `sync_backlog_drops_total`
