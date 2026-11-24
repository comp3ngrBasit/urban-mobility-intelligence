#!/usr/bin/env bash
# Full reproducible run on a fresh machine (no Docker): verify -> ingest -> clean -> train -> analyze.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.pipeline verify
python -m src.pipeline all
echo "Done. Start the API with: uvicorn backend.main:app --port 8000"
