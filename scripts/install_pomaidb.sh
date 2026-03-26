#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-autocookie/pomaidb}"
VERSION="${1:-latest}"
DEST="${DEST:-/usr/local/bin}"

tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

if [[ "$VERSION" == "latest" ]]; then
  rel_url="https://api.github.com/repos/${REPO}/releases/latest"
  tag="$(curl -fsSL "$rel_url" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
else
  tag="$VERSION"
fi

if [[ -z "${tag:-}" ]]; then
  echo "Could not resolve release tag"
  exit 1
fi

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$arch" in
  x86_64) arch="x64" ;;
  aarch64|arm64) arch="arm64" ;;
esac

asset="pomaidb-${os}-${arch}.tar.gz"
base="https://github.com/${REPO}/releases/download/${tag}"

curl -fsSL "${base}/${asset}" -o "${tmp}/${asset}"
if curl -fsSL "${base}/${asset}.sha256" -o "${tmp}/${asset}.sha256"; then
  (cd "$tmp" && sha256sum -c "${asset}.sha256")
fi

tar -xzf "${tmp}/${asset}" -C "$tmp"
install -m 0755 "${tmp}/pomaidb_server" "${DEST}/pomaidb_server"
install -m 0755 "${tmp}/pomaictl" "${DEST}/pomaictl" || true
echo "Installed pomaidb_server to ${DEST}"
