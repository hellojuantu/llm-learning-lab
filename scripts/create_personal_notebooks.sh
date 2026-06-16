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

这个目录放个人学习记录，不会提交到 git。

学习方式：

1. 从这里打开 `*.barry.ipynb`
2. 随便写笔记、运行 cell、改答案
3. 对外分享时，分享 `micrograd/playground/` 里的干净课程版

EOF
