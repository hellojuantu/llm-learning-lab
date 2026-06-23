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

Study note:

- `05_neuron_and_mlp.barry.ipynb` starts with a small linear algebra warm-up before Neuron / Layer / MLP: vector, dot product, bias, and why `sum(wi*xi) + b` is the neuron formula.
- It also includes a sketch for separating math brackets from Python lists: math `[]` shows vector/matrix shape, while Python `[]` stores objects.
- `07_classification_boundary.barry.ipynb` through `10_mini_project.barry.ipynb` extend the course after the basic training loop: classification, required PyTorch bridge, gradient debugging, and a final mini project.
- `08_pytorch_bridge.barry.ipynb` requires PyTorch. Install it before starting that notebook.

EOF
