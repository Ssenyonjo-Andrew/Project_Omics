#!/usr/bin/env bash
set -euo pipefail
cat data/raw/expanded/*.fasta > data/processed/sequences_expanded.fasta
echo "sequences: $(grep -c '^>' data/processed/sequences_expanded.fasta)"
cd-hit-est -i data/processed/sequences_expanded.fasta \
           -o data/processed/sequences_expanded_90.fasta \
           -c 0.90 -n 8 -M 0 -T 0
