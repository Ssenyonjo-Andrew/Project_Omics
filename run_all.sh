#!/usr/bin/env bash
set -euo pipefail
echo "=== 00 resolve taxa ==="; bash scripts/00_resolve_taxa.sh
echo "=== 01 expand ===";      python scripts/01_expand.py
echo "=== 02 cluster ===";     bash scripts/02_cluster.sh
echo "=== 03 split ===";       python scripts/03_split.py
echo "=== 04 verify ===";      bash scripts/04_verify.sh
echo "Phase 2 complete. Run 05_train.py next."
