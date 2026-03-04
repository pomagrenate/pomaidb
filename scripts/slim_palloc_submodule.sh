#!/usr/bin/env bash
# Configure sparse checkout for the palloc submodule to exclude large/unneeded
# directories (media ~5.7MB, test, bench) for smaller clone size on embedded.
# Run from repo root after: git submodule update --init third_party/palloc
set -e
PALLOC_DIR="${1:-third_party/palloc}"
if [[ ! -d "$PALLOC_DIR/.git" ]]; then
  echo "Not a git submodule: $PALLOC_DIR (run: git submodule update --init third_party/palloc)"
  exit 1
fi
cd "$PALLOC_DIR"
git config core.sparseCheckout true
mkdir -p .git/info
cat > .git/info/sparse-checkout << 'EOF'
# Include only what pomaidb needs to build (excludes media, test, bench, contrib)
/src
/include
/cmake
/CMakeLists.txt
/CMakePresets.json
/LICENSE
/README.md
/SECURITY.md
/palloc.pc.in
/azure-pipelines.yml
EOF
git read-tree -mu HEAD
echo "Sparse checkout configured for $PALLOC_DIR (media, test, bench, contrib excluded)."
cd - >/dev/null
