# Support Policy

## Versioning
- Semantic versioning (`MAJOR.MINOR.PATCH`).
- Breaking API or storage changes require `MAJOR` bump.

## Support Window
- Latest minor in current major: full support.
- Previous minor in current major: security + critical fixes.

## Compatibility Commitments
- `/v1` API remains backward-compatible across minor/patch releases.
- On-disk format changes must include migration notes and tests.
- SDKs target parity with the current major release line.
