#!/usr/bin/env python3
"""Fetch a diverse pool of 12 RNA and DNA viruses."""
import json, subprocess, time
from pathlib import Path

OUT = Path("data/raw/ncbi")
OUT.mkdir(parents=True, exist_ok=True)

PLAN = [
    # RNA Viruses
    ("dengue",      "Orthoflavivirus denguei",                 1000),
    ("chikungunya", "Chikungunya virus",                       1000),
    ("norovirus",   "Norovirus",                               1000),
    ("measles",     "Measles morbillivirus",                   1000),
    ("rabies",      "Rabies lyssavirus",                       1000),
    ("ebola",       "Zaire ebolavirus",                        1000),
    ("zika",        "Zika virus",                              1000),
    # DNA Viruses
    ("mpox",        "Monkeypox virus",                         1000),
    ("hsv1",        "Human alphaherpesvirus 1",                 500),
    ("adeno_c",     "Human mastadenovirus C",                   500),
    ("hpv16",       "Human papillomavirus 16",                 1000),
    ("hbv",         "Hepatitis B virus",                       1000),
]

def fetch(label, taxon, limit):
    jsonl = OUT / f"{label}.jsonl"
    fasta = OUT / f"{label}.fasta"
    
    if not jsonl.exists() or jsonl.stat().st_size == 0:
        print(f"  [{label}] probing {taxon}...")
        try:
            with open(jsonl, "w") as f:
                subprocess.run([
                    "datasets", "summary", "virus", "genome", "taxon", taxon,
                    "--complete-only", "--limit", str(limit), "--as-json-lines"
                ], stdout=f, stderr=subprocess.DEVNULL, check=True, timeout=3600)
        except subprocess.CalledProcessError:
            print(f"  [{label}] FAILED TO FIND TAXON: {taxon}")
            return
    
    n = sum(1 for _ in open(jsonl))
    print(f"  [{label}] {n} records found")

    if fasta.exists() and fasta.stat().st_size > 0:
        print(f"  [{label}] fasta cached")
        return

    accs = []
    for line in open(jsonl):
        try:
            a = json.loads(line).get("accession")
            if a: accs.append(a)
        except Exception: pass

    print(f"  [{label}] downloading {len(accs)} sequences...")
    with open(fasta, "w") as f:
        for i in range(0, len(accs), 100):
            try:
                r = subprocess.run([
                    "efetch", "-db", "nuccore", "-id", ",".join(accs[i:i+100]), "-format", "fasta"
                ], capture_output=True, text=True, timeout=600)
                if r.returncode == 0: f.write(r.stdout)
            except subprocess.TimeoutExpired: pass
            time.sleep(0.4)
    n_seq = sum(1 for l in open(fasta) if l.startswith(">"))
    print(f"  [{label}] saved {n_seq} sequences")

if __name__ == "__main__":
    for label, taxon, limit in PLAN:
        print(f"\n=== {label} ===")
        fetch(label, taxon, limit)
