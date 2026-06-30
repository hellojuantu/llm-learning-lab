#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found: $VENV_DIR"
  echo "Create it first with: uv venv $VENV_DIR"
  exit 1
fi

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

exec "$VENV_DIR/bin/python" -m notebook
