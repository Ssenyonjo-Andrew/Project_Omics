#!/usr/bin/env python3
"""Final split: 5-virus lineage holdout with MMseqs2 leakage check."""
import subprocess, tempfile, shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from Bio import SeqIO

RAW      = Path("data/raw/ncbi")
FILTERED = Path("data/processed/filtered")
LINEAGE  = Path("data/processed/lineage")
SPLIT    = Path("data/processed/split")
SPLIT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "dengue": {
        "fasta": FILTERED / "dengue.fasta",
        "lineage_csv": LINEAGE / "dengue_lineages.csv",
        "ref_len": 10663,
        "test": ["DENV-4"],
        "train_cap": 3000,
    },
    "chikungunya": {
        "fasta": FILTERED / "chikungunya.fasta",
        "lineage_csv": LINEAGE / "chikungunya_lineages.csv",
        "ref_len": 11764,
        "test": ["Asian"],
        "train_cap": 1500,
    },
    "hcv": {
        "fasta": RAW / "hcv.fasta",
        "lineage_csv": LINEAGE / "hcv_lineages.csv",
        "ref_len": 9375,
        "test": ["GT6"],
        "train_cap": 500,
    },
    "norovirus": {
        "fasta": RAW / "norovirus.fasta",
        "lineage_csv": LINEAGE / "norovirus_lineages.csv",
        "ref_len": 7536,
        "test": ["GI"],
        "train_cap": 1700,
    },
    "mpox": {
        "fasta": RAW / "mpox.fasta",
        "lineage_csv": LINEAGE / "mpox_lineages.csv",
        "ref_len": 196415,
        "test": ["IIb"],
        "train_cap": 400,
    },
}

VAL_FRAC = 0.15
TOLERANCE = 0.20
SEED = 42


def filter_and_label(name, cfg):
    lin_df = pd.read_csv(cfg["lineage_csv"])
    lin_map = {row["accession"].split(".")[0]: row["lineage"]
               for _, row in lin_df.iterrows()}
    ref_len = cfg["ref_len"]
    lo = int(ref_len * (1 - TOLERANCE))
    hi = int(ref_len * (1 + TOLERANCE))
    kept = {}
    for rec in SeqIO.parse(cfg["fasta"], "fasta"):
        base = rec.id.split(".")[0]
        if not (lo <= len(rec.seq) <= hi):
            continue
        lin = lin_map.get(base, "unknown")
        if lin == "unknown":
            continue
        kept[rec.id] = (rec, lin)
    return kept


def split_one(name, cfg):
    print(f"\n=== {name} ===")
    kept = filter_and_label(name, cfg)
    print(f"  after filter+label: {len(kept)}")
    if len(kept) < 30:
        print(f"  WARNING: too few sequences ({len(kept)}), skipping {name}")
        return None

    test_classes = set(cfg["test"])
    test  = {k: v for k, v in kept.items() if v[1] in test_classes}
    train = {k: v for k, v in kept.items() if v[1] not in test_classes}

    cap = cfg["train_cap"]
    if len(train) > cap:
        rng = np.random.RandomState(SEED)
        keys = list(train.keys()); rng.shuffle(keys)
        train = {k: train[k] for k in keys[:cap]}
        print(f"  train capped to {cap}")

    rng = np.random.RandomState(SEED + 1)
    train_keys = list(train.keys()); rng.shuffle(train_keys)
    n_val = max(1, int(round(len(train_keys) * VAL_FRAC)))
    val = {k: train[k] for k in train_keys[:n_val]}
    train = {k: train[k] for k in train_keys[n_val:]}

    for split_name, d in [("train", train), ("val", val), ("test", test)]:
        path = SPLIT / f"{name}_{split_name}.fasta"
        with open(path, "w") as f:
            for rec, _ in d.values():
                f.write(f">{rec.id}\n{rec.seq}\n")

    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(f"  test classes:  {dict(Counter(v[1] for v in test.values()))}")
    print(f"  train classes: {dict(Counter(v[1] for v in train.values()))}")
    if len(test) < 30:
        print(f"  WARNING: test set has only {len(test)} sequences")

    return {
        "name": name,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_lin": Counter(v[1] for v in test.values()),
        "train_lin": Counter(v[1] for v in train.values()),
    }


def combine_and_leakcheck(results):
    print("\n=== combining into unified train/val/test ===")
    for split in ["train", "val", "test"]:
        with open(SPLIT / f"all_{split}.fasta", "w") as out:
            for r in results:
                src = SPLIT / f"{r['name']}_{split}.fasta"
                if src.exists():
                    out.write(src.read_text())
        n = sum(1 for l in open(SPLIT / f"all_{split}.fasta") if l.startswith(">"))
        print(f"  all_{split}.fasta: {n} sequences")

    print("\n=== MMseqs2 leakage check (test vs train) ===")
    tmp = Path(tempfile.mkdtemp())
    try:
        result_tsv = tmp / "leak.tsv"
        subprocess.run([
            "mmseqs", "easy-search",
            str(SPLIT / "all_test.fasta"),
            str(SPLIT / "all_train.fasta"),
            str(result_tsv),
            str(tmp / "tmp"),
            "--search-type", "3",
            "--min-seq-id", "0.95",
            "-c", "0.8",
            "--cov-mode", "0",
            "--threads", "4",
        ], check=True, capture_output=True, text=True, timeout=3600)

        n_hits = 0
        hit_queries = set()
        if result_tsv.exists():
            for line in open(result_tsv):
                if line.strip():
                    n_hits += 1
                    hit_queries.add(line.split("\t")[0])
        n_test = sum(1 for l in open(SPLIT / "all_test.fasta") if l.startswith(">"))
        print(f"  test sequences: {n_test}")
        print(f"  matches at >=95%: {n_hits}")
        print(f"  affected test sequences: {len(hit_queries)}")
        if n_hits == 0:
            print("  PASS - zero leakage")
        else:
            print(f"  WARN - {len(hit_queries)}/{n_test} test sequences have a train match")
            for q in sorted(hit_queries)[:10]:
                print(f"    {q}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    results = []
    for name, cfg in CONFIG.items():
        r = split_one(name, cfg)
        if r:
            results.append(r)
    combine_and_leakcheck(results)

    print("\n=== SUMMARY ===")
    total_tr = sum(r["n_train"] for r in results)
    total_va = sum(r["n_val"] for r in results)
    total_te = sum(r["n_test"] for r in results)
    print(f"  train: {total_tr}")
    print(f"  val:   {total_va}")
    print(f"  test:  {total_te}")
    print(f"  total: {total_tr + total_va + total_te}")

    pd.DataFrame([{
        "virus": r["name"],
        "train": r["n_train"], "val": r["n_val"], "test": r["n_test"],
    } for r in results]).to_csv(SPLIT / "summary.csv", index=False)
    print(f"\nWrote {SPLIT}/summary.csv")


if __name__ == "__main__":
    main()
