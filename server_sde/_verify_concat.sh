#!/bin/bash
# Verify only completed runs with schema-2 provenance. Legacy CSVs need re-inference.
# Optional --repair reconstructs a damaged aggregate from verified individual outputs.
set -euo pipefail
exec "${PREDICT_PYTHON:-python3}" "$(dirname "$0")/_prediction_integrity.py" "$@"
