#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_PATH="${POMAI_LIB:-$ROOT_DIR/build/libpomai_c.so}"

if [[ ! -f "$LIB_PATH" ]]; then
  echo "Missing shared library: $LIB_PATH" >&2
  exit 1
fi

HAS_NUMPY="$(python3 - <<'PY'
import importlib.util
print(1 if importlib.util.find_spec("numpy") else 0)
PY
)"

if [[ "$HAS_NUMPY" == "1" ]]; then
  python3 "$ROOT_DIR/tools/pomai_bench.py" --lib "$LIB_PATH" recall
  python3 "$ROOT_DIR/tools/pomai_bench.py" --lib "$LIB_PATH" mixed-load --dim 256 --count 50000 --shards 2 --batch-size 512 --queries 500
else
  echo "numpy not available; running CI-sized recall matrix and mixed-load smoke."
  python3 "$ROOT_DIR/tools/pomai_bench.py" --lib "$LIB_PATH" recall --matrix ci
  python3 "$ROOT_DIR/tools/pomai_bench.py" --lib "$LIB_PATH" mixed-load --dim 128 --count 10000 --shards 2 --batch-size 256 --queries 200
fi
