# Edge Gateway Error Taxonomy

This document defines stable machine-readable error codes for `/v1`.

## Envelope
Every JSON error response should include:
- `status`: `error`
- `code`: machine-readable error code
- `message`: human-readable description

## Auth & Access
- `auth_scope_denied`: token missing required scope
- `mtls_required`: expected proxy mTLS header missing
- `not_found`: endpoint or resource not found

## Input Validation
- `invalid_path`: malformed URL path parameters
- `bad_vector`: vector payload parse/shape error
- `bad_value`: scalar payload parse error
- `bad_tokens`: token sequence parse error
- `bad_request`: generic request contract violation

## Write Path
- `write_failed`: backend write operation failed
- `durable_ack`: accepted with durability flush requested
- `accepted`: accepted without durability flush

## Idempotency & Rate
- `idempotency_key_seen`: duplicate write by idempotency key
- `rate_limited`: request rejected by rate limiter

## Lifecycle
- `expired_ttl`: record not visible due to TTL expiry
- `retention_count`: record evicted by count policy
- `retention_bytes`: record evicted by bytes policy
