"""Re-split: 3 test species (3 families), val from training species."""
import yaml, numpy as np, pandas as pd
from pathlib import Path

cfg = yaml.safe_load(open("config.yaml"))
SEED = cfg["project"]["seed"]
VAL_FRAC = cfg["split"]["val_fraction"]
TEST = set(cfg["split"]["test_species"])
TRAIN_POOL = set(cfg["split"]["train_species"])

df = pd.read_csv("data/processed/master_expanded.csv")
def slug(path): return Path(path).stem
df["slug"] = df["sequence_file_path"].apply(slug)

# Clusters
clusters = {}
cid = None
for line in open("data/processed/sequences_expanded_90.fasta.clstr"):
    if line.startswith(">Cluster"): cid = line.split()[1]
    elif cid: clusters[line.split(">")[1].split(".")[0]] = f"C{cid}"
df["accession_base"] = df["accession"].str.split(".").str[0]
df["cluster_id"] = df["accession_base"].map(clusters)

rng = np.random.default_rng(SEED)
rows = []
for s in df["slug"].unique():
    sub = df[df["slug"] == s].copy()
    if s in TEST:
        sub["split"] = "test"
    elif s in TRAIN_POOL:
        idx = rng.permutation(len(sub))
        n_val = int(round(len(sub) * VAL_FRAC))
        sub["split"] = "train"
        sub.loc[sub.index[idx[:n_val]], "split"] = "val"
    else:
        raise ValueError(f"species {s} not in any split list")
    rows.append(sub)

df = pd.concat(rows).drop(columns=["slug", "accession_base"])
df = df[["accession","sequence_file_path","virus_name","family","genus","species","cluster_id","split"]]
df.to_csv("data/processed/master_final.csv", index=False)

print(df["split"].value_counts().to_string())
print("\nper-species:")
print(pd.crosstab(df["virus_name"], df["split"]).to_string())
print("\nper-family in test:")
print(df[df.split=="test"]["family"].value_counts().to_string())
