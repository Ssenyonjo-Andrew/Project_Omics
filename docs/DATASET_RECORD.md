# Dataset Record — Multi-Family Viral Lineage Classifier

- Version: 1.0.0
- Status: in progress
- Created: 2026-10-29

## Scope
- 9 viruses across 5 families
- Classification target: virus lineage (~32 classes)
- Test holdouts: DENV-4, WNV-L2, YF-East-African, Zaire EBOV,
  Marburg clade B, Rabies Africa-2, RVFV H/K/M, CCHF Asia-1,
  Chikungunya Asian

## Sources
- Sequences: NCBI Nucleotide (Entrez Direct)
- Taxonomy: ICTV VMR (pin version)
- Annotations: NCBI GenBank feature tables

## Clustering
- Tool: MMseqs2 all-vs-all
- Thresholds: 90% and 95%

## Split
- Strategy: lineage-level holdout
- Train: all non-holdout lineages + 10% validation
- Test: holdout lineages only

## Leakage
- Check: mmseqs2 easy-search test vs train, 95% identity
- Result: <pending>
- Status: <pending>

## Sequence counts
| Virus | Train | Val | Test | Total |
|---|---|---|---|---|
| Dengue | | | | |
| West Nile | | | | |
| Yellow fever | | | | |
| Ebola | | | | |
| Marburg | | | | |
| Rabies | | | | |
| RVFV | | | | |
| CCHF | | | | |
| Chikungunya | | | | |
