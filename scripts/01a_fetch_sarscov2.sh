#!/usr/bin/env bash
set -uo pipefail
SLUG=sars_cov_2
OUT_JSON=data/raw/expanded/${SLUG}.jsonl
OUT_FA=data/raw/expanded/${SLUG}.fasta
CAP=87
SEED=42
PER_YEAR=18
YEARS="2020 2021 2022 2023 2024"

mkdir -p data/raw/expanded
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "[01a] SARS-CoV-2 stratified fetch (CAP=${CAP}, seed=${SEED})"

for year in $YEARS; do
  esearch -db nuccore \
    -query "txid2697049[Organism] AND \"complete genome\"[Title] AND ${year}[PDAT]" \
    2>/dev/null \
    | efetch -db nuccore -format acc 2>/dev/null \
    > ${TMP}/${year}_all.txt || true

  n=$(wc -l < ${TMP}/${year}_all.txt)
  echo "  ${year}: ${n} accessions available"

  if [ "$n" -eq 0 ]; then
    echo "    WARNING: no records for ${year}"
    continue
  fi

  python3 -c "
import random
random.seed(${SEED} + ${year})
with open('${TMP}/${year}_all.txt') as f:
    accs = [l.strip() for l in f if l.strip() and l.strip().startswith(('NC_','MN_','MW_','MZ_','OK_','OL_','OM_','OP_','OQ_','OR_','PP_','PQ_','PR_','PS_','PT_','PU_','PV_','PW_','PX_','PY_','PZ_')) or l.strip() and len(l.strip()) >= 6]
random.shuffle(accs)
with open('${TMP}/${year}_sampled.txt','w') as f:
    for acc in accs[:${PER_YEAR}]:
        f.write(acc + '\n')
print(f'    sampled {min(len(accs), ${PER_YEAR})}')
"
done

cat ${TMP}/*_sampled.txt > ${TMP}/all_sampled.txt 2>/dev/null || true
total=$(wc -l < ${TMP}/all_sampled.txt)
echo "  total sampled: ${total}"

if [ "$total" -lt "$CAP" ]; then
  echo "  WARNING: only ${total} sampled, less than CAP=${CAP}"
fi

echo "  fetching FASTA for ${total} accessions..."
ids_csv=$(paste -sd, ${TMP}/all_sampled.txt)
efetch -db nuccore -id "$ids_csv" -format fasta 2>/dev/null > ${TMP}/all.fasta || true

python3 -c "
from Bio import SeqIO
import random, json
random.seed(${SEED})
records = list(SeqIO.parse('${TMP}/all.fasta','fasta'))
random.shuffle(records)
selected = records[:${CAP}]
SeqIO.write(selected, '${OUT_FA}', 'fasta')
with open('${OUT_JSON}','w') as f:
    for r in selected:
        f.write(json.dumps({'accession': r.id})+'\n')
print(f'  wrote {len(selected)} sequences')
"

n=$(grep -c '^>' ${OUT_FA})
echo "  ${n} sequences -> ${OUT_FA}"
[ "$n" -ge "$CAP" ] || { echo "FAIL: only ${n} sequences, need ${CAP}"; exit 1; }
echo "[01a] PASS"