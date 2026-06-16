#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/micrograd/playground"
DST_DIR="$ROOT_DIR/micrograd/personal"

mkdir -p "$DST_DIR"

for src in "$SRC_DIR"/*.ipynb; do
  name="$(basename "$src" .ipynb)"
  dst="$DST_DIR/${name}.barry.ipynb"
  if [[ -e "$dst" ]]; then
    echo "skip existing: $dst"
  else
    cp "$src" "$dst"
    echo "created: $dst"
  fi
done

cat > "$DST_DIR/README.md" <<'EOF'
# Personal Notebooks

This directory is for personal study notes and is ignored by git.

How to use:

1. Open `*.barry.ipynb` from this directory.
2. Write notes, run cells, and edit answers freely.
3. When sharing with others, share the clean course version in `micrograd/playground/`.

EOF
