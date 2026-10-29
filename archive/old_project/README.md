# xAI Viral Attribution — Project Scaffold

## Quick start
    conda activate virgo
    bash run_all.sh

## What each stage does
- `00_resolve_taxa.sh` — finds the working NCBI taxon name per species (solves the SARS-CoV-2 alias problem)
- `01_expand.py` — balanced, stratified download; CAP=87/species; asserts balance
- `02_cluster.sh` — CD-HIT-EST at 90%
- `03_split.py` — species-level train/val/test
- `04_verify.sh` — cd-hit-2d leak check; exits non-zero on any match

## Phase 3/4
`05_train.py`, `06_attribute.py`, `07_evaluate.py` are scaffolds with the
data-loading and checkpointing conventions fixed. The model bodies and the
attribution implementations are yours to write. This is deliberate: the
supervisor expects you to have written the model code.

## Hard rules
- Every species contributes exactly 87 sequences
- Leak check must print PASS before any training
- All thresholds in config.yaml; do not hardcode them in scripts
