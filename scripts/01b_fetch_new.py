#!/usr/bin/env python3
"""Fetch the 5 new viruses for the final 7-virus set."""
import json, subprocess, time
from pathlib import Path

OUT = Path("data/raw/ncbi")
OUT.mkdir(parents=True, exist_ok=True)

PLAN = [
    ("hiv1",      "Human immunodeficiency virus 1", 5000),
    ("hcv",       "Hepatitis C virus",              2000),
    ("hbv",       "Hepatitis B virus",              5000),
    ("norovirus", "Norovirus",                      3000),
    ("mpox",      "Monkeypox virus",                5000),
]


def fetch(label, taxon, limit):
    jsonl = OUT / f"{label}.jsonl"
    fasta = OUT / f"{label}.fasta"
    if jsonl.exists() and jsonl.stat().st_size > 0:
        n = sum(1 for _ in open(jsonl))
        print(f"  [{label}] summary cached ({n})")
    else:
        print(f"  [{label}] probing {taxon} (limit {limit})")
        with open(jsonl, "w") as f:
            subprocess.run([
                "datasets", "summary", "virus", "genome", "taxon", taxon,
                "--complete-only", "--limit", str(limit), "--as-json-lines"
            ], stdout=f, stderr=subprocess.PIPE, check=True, timeout=3600)
        n = sum(1 for _ in open(jsonl))
        print(f"  [{label}] {n} records")

    if fasta.exists() and fasta.stat().st_size > 0:
        n = sum(1 for l in open(fasta) if l.startswith(">"))
        print(f"  [{label}] fasta cached ({n})")
        return n

    accs = []
    for line in open(jsonl):
        try:
            a = json.loads(line).get("accession")
            if a: accs.append(a)
        except Exception:
            pass

    print(f"  [{label}] downloading {len(accs)} via efetch")
    with open(fasta, "w") as f:
        for i in range(0, len(accs), 100):
            try:
                r = subprocess.run([
                    "efetch", "-db", "nuccore",
                    "-id", ",".join(accs[i:i+100]), "-format", "fasta"
                ], capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    f.write(r.stdout)
            except subprocess.TimeoutExpired:
                print(f"    batch {i//100+1} timeout")
            time.sleep(0.4)
            if (i//100 + 1) % 10 == 0:
                print(f"    ... {i+100}/{len(accs)}")
    n = sum(1 for l in open(fasta) if l.startswith(">"))
    print(f"  [{label}] {n} downloaded")
    return n


def main():
    print("Fetching 5 new viruses for 7-virus final set\n")
    for label, taxon, limit in PLAN:
        print(f"=== {label}: {taxon} ===")
        fetch(label, taxon, limit)
        print()


if __name__ == "__main__":
    main()
