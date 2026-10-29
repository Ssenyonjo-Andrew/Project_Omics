"""Per-species cluster split at escalating identity thresholds.
Ensures zero leakage by construction (whole clusters held out)."""
import subprocess, tempfile, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio import SeqIO

MASTER_IN = "data/processed/master_expanded.csv"
FASTA_IN  = "data/processed/sequences_expanded.fasta"
OUT_CSV   = "data/processed/master_cluster.csv"
OUT_TRAIN = "data/processed/train_cluster.fasta"
OUT_VAL   = "data/processed/val_cluster.fasta"
OUT_TEST  = "data/processed/test_cluster.fasta"

IDENTITY_LEVELS = [0.95, 0.98, 0.99, 0.995, 0.999]
TRAIN_TARGET = 0.75
VAL_TARGET   = 0.10
SEED = 42


def run_cdhit(fasta_path, identity, out_prefix):
    subprocess.run([
        "cd-hit-est", "-i", str(fasta_path), "-o", str(out_prefix),
        "-c", str(identity), "-n", "8", "-M", "0", "-T", "0"
    ], check=True, capture_output=True, text=True, timeout=600)
    clusters, cid = {}, None
    for line in open(str(out_prefix) + ".clstr"):
        line = line.strip()
        if line.startswith(">Cluster"):
            cid = int(line.split()[1])
        elif line and cid is not None:
            acc = line.split(">")[1].split(".")[0]
            clusters[acc] = cid
    return clusters


def assign_by_target(cluster_sizes, targets):
    sorted_cids = sorted(cluster_sizes.keys(), key=lambda c: -cluster_sizes[c])
    running = {k: 0 for k in targets}
    assignment = {}
    for cid in sorted_cids:
        deficits = {s: targets[s] - running[s] for s in targets}
        best = max(deficits, key=lambda k: (deficits[k], -list(targets).index(k)))
        assignment[cid] = best
        running[best] += cluster_sizes[cid]
    return assignment, running


def force_non_empty(assignment, running, cluster_sizes):
    for split in ["val", "test"]:
        if running[split] == 0 and len(assignment) >= 3:
            largest_split = max(running, key=running.get)
            candidates = [c for c, s in assignment.items() if s == largest_split]
            if candidates:
                smallest = min(candidates, key=lambda c: cluster_sizes[c])
                assignment[smallest] = split
                running[largest_split] -= cluster_sizes[smallest]
                running[split] = cluster_sizes[smallest]
    return assignment, running


def main():
    df = pd.read_csv(MASTER_IN)
    df["slug"] = df["sequence_file_path"].apply(lambda p: Path(p).stem)

    seqs = {}
    for rec in SeqIO.parse(FASTA_IN, "fasta"):
        seqs[rec.id.split(".")[0]] = (rec.id, str(rec.seq))

    all_splits = {}
    tmpdir = tempfile.mkdtemp()
    try:
        for slug in sorted(df["slug"].unique()):
            sub = df[df["slug"] == slug]
            accs = [a.split(".")[0] for a in sub["accession"]]
            n_seq = len(accs)

            sp_fa = Path(tmpdir) / f"{slug}.fasta"
            with open(sp_fa, "w") as f:
                for acc in accs:
                    if acc in seqs:
                        rid, s = seqs[acc]
                        f.write(f">{rid}\n{s}\n")

            clusters, used_id = None, None
            for ident in IDENTITY_LEVELS:
                try:
                    c = run_cdhit(sp_fa, ident, Path(tmpdir) / f"{slug}_{int(ident*1000)}")
                except subprocess.CalledProcessError:
                    continue
                if len(set(c.values())) >= 3:
                    clusters, used_id = c, ident
                    break

            if clusters is None:
                print(f"  {slug:15s} could not reach 3 clusters - all to train")
                for acc in accs:
                    all_splits[acc] = "train"
                continue

            by_c = defaultdict(int)
            for cid in clusters.values():
                by_c[cid] += 1
            cluster_sizes = dict(by_c)

            targets = {
                "train": max(1, int(round(n_seq * TRAIN_TARGET))),
                "val":   max(1, int(round(n_seq * VAL_TARGET))),
                "test":  0,
            }
            targets["test"] = n_seq - targets["train"] - targets["val"]
            if targets["test"] < 1:
                targets["test"] = 1
                targets["train"] = n_seq - targets["val"] - 1

            assignment, running = assign_by_target(cluster_sizes, targets)
            assignment, running = force_non_empty(assignment, running, cluster_sizes)

            for acc, cid in clusters.items():
                all_splits[acc] = assignment[cid]

            sizes_top = sorted(cluster_sizes.values(), reverse=True)[:5]
            print(f"  {slug:15s} id={used_id*100:.1f}%  clusters={len(cluster_sizes):2d} "
                  f"(top sizes: {sizes_top})  "
                  f"actual: train={running['train']} val={running['val']} test={running['test']}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    df["cluster_split"] = df["accession"].str.split(".").str[0].map(all_splits)
    missing = df["cluster_split"].isna().sum()
    if missing:
        print(f"\nWARNING: {missing} sequences unassigned -> train")
        df.loc[df["cluster_split"].isna(), "cluster_split"] = "train"

    df.to_csv(OUT_CSV, index=False)

    for split_name, path in [("train", OUT_TRAIN), ("val", OUT_VAL), ("test", OUT_TEST)]:
        with open(path, "w") as f:
            for _, row in df[df["cluster_split"] == split_name].iterrows():
                acc = row["accession"].split(".")[0]
                if acc in seqs:
                    rid, s = seqs[acc]
                    f.write(f">{rid}\n{s}\n")

    print(f"\n=== SPLIT TOTALS ===")
    print(df["cluster_split"].value_counts().to_string())
    print(f"\n=== PER-SPECIES x SPLIT ===")
    print(pd.crosstab(df["virus_name"], df["cluster_split"]).to_string())
    print(f"\n=== FASTA counts ===")
    for path in [OUT_TRAIN, OUT_VAL, OUT_TEST]:
        n = sum(1 for l in open(path) if l.startswith(">"))
        print(f"  {path}: {n}")

    print(f"\n=== leakage check at 95% ===")
    try:
        subprocess.run([
            "cd-hit-est-2d", "-i", OUT_TEST, "-i2", OUT_TRAIN,
            "-o", "data/processed/test_vs_train_cluster.fasta",
            "-c", "0.95", "-n", "8", "-M", "0", "-T", "0"
        ], check=True, capture_output=True, text=True, timeout=600)

        n_leak = sum(1 for l in open("data/processed/test_vs_train_cluster.fasta.clstr")
                     if l.startswith(">Cluster"))
        n_test = sum(1 for l in open(OUT_TEST) if l.startswith(">"))
        print(f"  Test sequences: {n_test}")
        print(f"  Contaminated at 95%: {n_leak}")
        if n_leak == 0:
            print(f"  PASS - zero leakage")
        else:
            print(f"  WARN - {100*n_leak/n_test:.0f}% of test contaminated")
    except subprocess.CalledProcessError as e:
        print(f"  leakage check failed: {e}")


if __name__ == "__main__":
    main()
