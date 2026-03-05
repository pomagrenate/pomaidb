#!/usr/bin/env bash
# CI / policy check: PomaiDB core must not use raw malloc or new.
# All aligned allocation must go through palloc_compat.h (palloc_malloc_aligned /
# palloc_free) or placement new with palloc_malloc_aligned. Excludes third_party.
# Run from repo root.
set -e
ROOT="${1:-.}"
cd "$ROOT"
FAIL=0
for dir in src include; do
  if [[ ! -d "$dir" ]]; then continue; fi
  if grep -rn --include='*.cc' --include='*.cpp' --include='*.c' --include='*.h' --include='*.hpp' -E '\bmalloc\s*\(' "$dir" 2>/dev/null; then
    echo "error: $dir contains malloc() - use palloc_compat.h (palloc_malloc_aligned) instead"
    FAIL=1
  fi
  # Match " new " (operator new / new Type) but not placement new "new (", comments, or log strings
  if grep -rn --include='*.cc' --include='*.cpp' --include='*.h' --include='*.hpp' -E '\bnew\s+[A-Za-z_:]' "$dir" 2>/dev/null | grep -v 'new\s*(' | grep -v '//.*new ' | grep -v 'POMAI_LOG' | grep -v '"created new '; then
    echo "error: $dir contains operator new - use palloc_compat.h + placement new only"
    FAIL=1
  fi
done
if [[ $FAIL -eq 1 ]]; then
  echo "Policy: no raw malloc/new in src/ or include/. Use palloc_compat.h and placement new."
  exit 1
fi
echo "check_no_malloc_new: ok (no raw malloc/new in src, include)"
