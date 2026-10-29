#!/usr/bin/env bash
# Tries candidate taxon names for each species and records the working one.
set -euo pipefail
OUT="data/raw/resolved_taxa.tsv"
: > "$OUT"
probe() {
  local slug="$1"; shift
  for t in "$@"; do
    n=$(datasets summary virus genome taxon "$t" \
          --complete-only --limit 5 --as-json-lines 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
      printf "%-14s\t%s\t%s\n" "$slug" "$t" "$n" | tee -a "$OUT"
      return 0
    fi
  done
  printf "%-14s\tNONE\t0\n" "$slug" | tee -a "$OUT"
  return 1
}
probe dengue       "Dengue virus"
probe ebola        "Zaire ebolavirus"
probe mers         "Middle East respiratory syndrome coronavirus"
probe yellow_fever "Yellow fever virus"
probe zika         "Zika virus"
probe marburg      "Orthomarburgvirus marburgense" "Marburgvirus"
probe lassa        "Lassa virus" "Mammarenavirus lassaense"
probe nipah        "Nipah virus" "Henipavirus nipahense"
probe sars_cov_2   "SARS-CoV-2" "Severe acute respiratory syndrome coronavirus 2" "Human coronavirus 2019" "Betacoronavirus pandemicum"
echo "resolved -> $OUT"
