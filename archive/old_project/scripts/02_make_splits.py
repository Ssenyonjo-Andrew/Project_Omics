import pandas as pd
import numpy as np

LABELS_IN   = "data/processed/labels.csv"
CLSTR_IN    = "data/processed/sequences_90.fasta.clstr"
MASTER_OUT  = "data/processed/master.csv"
SPLITS_OUT  = "data/processed/splits.csv"

SEED          = 42
N_VAL_SPECIES = 2
N_TEST_SPECIES= 2

def parse_clstr(path):
    clusters, cid = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">Cluster"):
            cid = line.split()[1]
        elif line:
            acc = line.split(">")[1].split(".")[0]
            clusters[acc] = f"C{cid}"
    return clusters

def main():
    labels = pd.read_csv(LABELS_IN)
    labels["accession_base"] = labels["accession"].str.split(".").str[0]

    species = sorted(labels["species"].unique())
    if len(species) != 10:
        print(f"WARNING: expected 10 species, found {len(species)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(species)

    test_species  = species[:N_TEST_SPECIES]
    val_species   = species[N_TEST_SPECIES:N_TEST_SPECIES+N_VAL_SPECIES]
    train_species = species[N_TEST_SPECIES+N_VAL_SPECIES:]

    print(f"Train ({len(train_species)}): {train_species}")
    print(f"Val   ({len(val_species)}): {val_species}")
    print(f"Test  ({len(test_species)}): {test_species}")

    def assign(sp):
        if sp in test_species: return "test"
        if sp in val_species:  return "val"
        return "train"

    labels["split"] = labels["species"].apply(assign)

    clusters = parse_clstr(CLSTR_IN)
    labels["cluster_id"] = labels["accession_base"].map(clusters)

    # Final master schema (exact column order)
    master = labels[[
        "accession", "virus_name", "family", "genus",
        "species", "cluster_id", "split"
    ]].copy()
    master["sequence_file_path"] = "data/processed/sequences.fasta"
    master = master[[
        "accession", "sequence_file_path", "virus_name",
        "family", "genus", "species", "cluster_id", "split"
    ]]

    master.to_csv(MASTER_OUT, index=False)
    labels[["accession", "split"]].to_csv(SPLITS_OUT, index=False)

    print(f"\nSplit counts:")
    print(master["split"].value_counts().to_string())
    print(f"\nSequences per split per virus:")
    print(pd.crosstab(master["virus_name"], master["split"]).to_string())
    print(f"\nSaved: {MASTER_OUT}")
    print(f"Saved: {SPLITS_OUT}")

if __name__ == "__main__":
    main()