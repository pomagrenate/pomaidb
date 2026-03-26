#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC="${ROOT}/api/openapi/v1.yaml"
TAX="${ROOT}/docs/API_ERROR_TAXONOMY.md"

test -f "$SPEC"
test -f "$TAX"

if ! rg -q "^openapi:" "$SPEC"; then
  echo "OpenAPI spec missing header"
  exit 1
fi

if ! rg -q "auth_scope_denied" "$TAX"; then
  echo "Error taxonomy missing required code"
  exit 1
fi

echo "API contract files present and minimally valid."
