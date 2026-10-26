#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/master_final.csv")
fa = "data/processed/sequences_expanded.fasta"
for sp in ("train","test"):
    keep = set(df[df.split==sp]["accession"].str.split(".").str[0])
    out, w = [], False
    for line in open(fa):
        if line.startswith(">"):
            w = line[1:].split(".")[0] in keep
        if w: out.append(line)
    open(f"data/processed/{sp}.fasta","w").writelines(out)
    print(f"{sp}: {sum(1 for l in out if l.startswith('>'))} seqs")
PY
out=$(cd-hit-est-2d -i data/processed/test.fasta -i2 data/processed/train.fasta \
      -o data/processed/test_vs_train.fasta -c 0.90 -n 8 -M 0 -T 0 2>&1)
n=$(echo "$out" | grep -oP '\d+(?= +clustered)' | tail -1)
echo "test-vs-train matches at 90%: ${n:-0}"
if [ "${n:-0}" = "0" ]; then echo "PASS - no leakage"; else echo "FAIL"; exit 1; fi
