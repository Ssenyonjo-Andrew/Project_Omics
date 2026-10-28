"""Per-species temporal quantile split (SARS-CoV-2 excluded)."""
import pandas as pd
import subprocess
from pathlib import Path
from Bio import SeqIO

MASTER_IN   = "data/processed/master_dated.csv"
MASTER_OUT  = "data/processed/master_temporal.csv"
FASTA_IN    = "data/processed/sequences_expanded.fasta"
FASTA_TRAIN = "data/processed/train_temporal.fasta"
FASTA_VAL   = "data/processed/val_temporal.fasta"
FASTA_TEST  = "data/processed/test_temporal.fasta"

EXCLUDED_SPECIES = {"sars_cov_2"}
TRAIN_Q = 0.70
VAL_Q   = 0.80
CDHIT_IDENTITY = 0.95


def extend_to_year_end(sub, idx):
    if idx >= len(sub) or idx == 0:
        return idx
    cutoff_year = sub.loc[idx - 1, "year"]
    while idx < len(sub) and sub.loc[idx, "year"] == cutoff_year:
        idx += 1
    return idx


def main():
    df = pd.read_csv(MASTER_IN)
    df["slug"] = df["sequence_file_path"].apply(lambda p: Path(p).stem)

    excluded = df[df["slug"].isin(EXCLUDED_SPECIES)].copy()
    excluded["split"] = "unused"

    kept = df[~df["slug"].isin(EXCLUDED_SPECIES)].copy()
    no_date = kept[kept["year"].isna()].copy()
    no_date["split"] = "no_date"
    dated = kept[kept["year"].notna()].copy()

    print(f"excluded (SARS-CoV-2): {len(excluded)} sequences")
    print(f"no date:               {len(no_date)} sequences")
    print(f"temporal study:        {len(dated)} sequences across "
          f"{dated['slug'].nunique()} species\n")

    rows = []
    for slug in sorted(dated["slug"].unique()):
        sub = dated[dated["slug"] == slug].copy()
        sub["_m"] = sub["month"].fillna(1).astype(int)
        sub["_d"] = sub["day"].fillna(1).astype(int)
        sub = sub.sort_values(["year", "_m", "_d", "accession"]).reset_index(drop=True)
        n = len(sub)

        train_cut = extend_to_year_end(sub, int(round(n * TRAIN_Q)))
        val_cut   = extend_to_year_end(sub, int(round(n * VAL_Q)))
        if val_cut < train_cut:
            val_cut = train_cut

        sub["split"] = "test"
        sub.loc[:val_cut - 1, "split"]   = "val"
        sub.loc[:train_cut - 1, "split"] = "train"
        sub = sub.drop(columns=["_m", "_d"])
        rows.append(sub)

        n_tr = (sub["split"] == "train").sum()
        n_va = (sub["split"] == "val").sum()
        n_te = (sub["split"] == "test").sum()
        med_tr = sub.loc[sub["split"] == "train", "year"].median()
        med_te = sub.loc[sub["split"] == "test",  "year"].median()
        print(f"  {slug:15s} train={n_tr:2d}(med {med_tr:.0f})  "
              f"val={n_va:2d}  test={n_te:2d}(med {med_te:.0f})")

    result = pd.concat(rows + [excluded, no_date], ignore_index=True)
    result = result.drop(columns=["slug"])
    result = result[[
        "accession", "sequence_file_path", "virus_name", "family",
        "genus", "species", "year", "month", "day",
        "date_source", "split"
    ]]
    result.to_csv(MASTER_OUT, index=False)

    print(f"\nWrote {MASTER_OUT}")
    print(f"\nSplit totals:")
    print(result["split"].value_counts().to_string())
    print(f"\nPer-virus × split:")
    print(pd.crosstab(result["virus_name"], result["split"]).to_string())

    # split FASTAs
    by_split = {s: set(result[result["split"] == s]["accession"].str.split(".").str[0])
                for s in ["train", "val", "test"]}
    paths = {"train": FASTA_TRAIN, "val": FASTA_VAL, "test": FASTA_TEST}
    handles = {s: open(paths[s], "w") for s in paths}
    try:
        for rec in SeqIO.parse(FASTA_IN, "fasta"):
            base = rec.id.split(".")[0]
            for split, accs in by_split.items():
                if base in accs:
                    handles[split].write(f">{rec.id}\n{rec.seq}\n")
                    break
    finally:
        for h in handles.values():
            h.close()
    for split, path in paths.items():
        n = sum(1 for l in open(path) if l.startswith(">"))
        print(f"  {path:45s} {n} sequences")

    # leakage check (temporal near-duplicates are expected; reported, not fatal)
    print(f"\n=== leakage check (cd-hit-est-2d at {CDHIT_IDENTITY*100:.0f}%) ===")
    r = subprocess.run([
        "cd-hit-est-2d",
        "-i", FASTA_TEST,
        "-i2", FASTA_TRAIN,
        "-o", "data/processed/test_vs_train_temporal.fasta",
        "-c", str(CDHIT_IDENTITY),
        "-n", "8", "-M", "0", "-T", "0",
    ], capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    matches = 0
    for line in out.splitlines():
        if "compared" in line and "clustered" in line:
            parts = line.split()
            matches = int(parts[2])
            print(f"  {line.strip()}")
            break
    if matches == 0:
        print(f"  PASS - zero test sequences match at >= {CDHIT_IDENTITY*100:.0f}%")
    else:
        print(f"  WARN - {matches} test sequence(s) match a train sequence "
              f"at >= {CDHIT_IDENTITY*100:.0f}%")
        print(f"  Temporal near-duplicates. Document in Chapter 3, do not re-split.")


if __name__ == "__main__":
    main()
