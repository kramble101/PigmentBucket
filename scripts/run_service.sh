#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

exec uvicorn src.service.app:app --host 127.0.0.1 --port 8765 --reload
