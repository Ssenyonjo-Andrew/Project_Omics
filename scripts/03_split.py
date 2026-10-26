"""Species-level split. Hardcoded species sets per config.yaml."""
from pathlib import Path
import yaml, pandas as pd

cfg = yaml.safe_load(open("config.yaml"))
clusters = {}
cid = None
for line in open("data/processed/sequences_expanded_90.fasta.clstr"):
    line = line.rstrip()
    if line.startswith(">Cluster"):
        cid = line.split()[1]
    elif cid is not None:
        clusters[line.split(">")[1].split(".")[0]] = f"C{cid}"

df = pd.read_csv("data/processed/master_expanded.csv")
df["accession_base"] = df["accession"].str.split(".").str[0]
df["cluster_id"] = df["accession_base"].map(clusters)

test  = set(cfg["split"]["test_species"])
val   = set(cfg["split"]["val_species"])
def assign(row):
    from pathlib import Path
    slug = Path(row["sequence_file_path"]).stem
    if slug in test:  return "test"
    if slug in val:   return "val"
    if slug in cfg["split"]["train_species"]: return "train"
    raise ValueError(f"slug {slug!r} not in config split lists")
df["split"] = df.apply(assign, axis=1)
df = df.drop(columns=["accession_base"])
df.to_csv("data/processed/master_final.csv", index=False)
print(df["split"].value_counts().to_string())
print("\nper-virus:")
print(pd.crosstab(df["virus_name"], df["split"]).to_string())
