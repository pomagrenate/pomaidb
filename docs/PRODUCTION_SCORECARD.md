# PomaiDB Production Readiness Scorecard

This scorecard is used for release-go/no-go decisions.

## 1. Security
- Artifact checksums are generated and published.
- Release artifacts are signed.
- SBOM is generated for release bundles.
- Dependency scanning runs in CI.

## 2. Reliability
- Full `ctest` pass on release branch.
- `ctest -L bench` pass on release branch.
- Crash/recovery tests pass.
- Gateway replay/idempotency tests pass.

## 3. Compatibility
- `/v1` API golden tests pass.
- On-disk format compatibility tests pass.
- SDK conformance tests pass across supported SDKs.

## 4. Operability
- Health endpoint returns grade + rationale.
- Metrics include sync lag/queue/drop visibility.
- Audit logs are structured and parsable.
- Runbooks exist for restore, replay, and retention incidents.

## 5. Portability
- Linux/macOS/Windows release artifacts built.
- Embedded and server mode smoke tests pass.
- Vulkan optional path documented and tested.

## Release Gate
A release is production-ready when all sections above are green and no P0/P1 issue is open.
