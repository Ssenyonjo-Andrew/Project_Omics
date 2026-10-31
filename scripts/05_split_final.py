#!/usr/bin/env python3
import random
from pathlib import Path
from collections import Counter
import pandas as pd
from Bio import SeqIO

RAW = Path("data/raw/ncbi")
OUT = Path("data/processed/final_split")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# Holdout rules (Adjusted for data availability)
TEST_HOLDOUTS = {
    "dengue": "DENV-4",
    "measles": "B3",
    "norovirus": "GI",
    "mpox": "IIb",
    "adeno_c": "Type_1",
}

def main():
    df = pd.read_csv("data/processed/lineage/master_lineage_6v.csv")
    lin_map = {(row["virus"], row["accession"].split(".")[0]): row["lineage"] for _, row in df.iterrows()}
    
    # Load the old clustered Chikungunya labels
    chik_df = pd.read_csv("data/processed/lineage/chikungunya_lineages.csv")
    chik_map = {row["accession"].split(".")[0]: row["lineage"] for _, row in chik_df.iterrows()}
    
    train_recs, test_recs = [], []
    
    for virus in ["dengue", "chikungunya", "norovirus", "measles", "mpox", "adeno_c"]:
        fasta = RAW / f"{virus}.fasta"
        v_train, v_test = [], []
        
        for rec in SeqIO.parse(fasta, "fasta"):
            base = rec.id.split(".")[0]
            
            # Get lineage (use old CSV for Chikungunya)
            if virus == "chikungunya":
                lin = chik_map.get(base, "unknown")
            else:
                lin = lin_map.get((virus, base), "unknown")
                
            if lin == "unknown": continue
            
            # Apply holdout rule
            holdout = TEST_HOLDOUTS.get(virus, "Asian") # Default to Asian for Chikungunya
            if lin == holdout:
                v_test.append(rec)
            else:
                v_train.append(rec)
                
        print(f"{virus:12s} | Train: {len(v_train):4d} | Test ({holdout}): {len(v_test):4d}")
        
        if len(v_train) > 800:
            random.seed(SEED)
            random.shuffle(v_train)
            v_train = v_train[:800]
            
        train_recs.extend([(virus, r) for r in v_train])
        test_recs.extend([(virus, r) for r in v_test])

    with open(OUT / "train.fasta", "w") as f:
        for v, rec in train_recs: f.write(f">{v}|{rec.id}\n{rec.seq}\n")
    with open(OUT / "test.fasta", "w") as f:
        for v, rec in test_recs: f.write(f">{v}|{rec.id}\n{rec.seq}\n")
        
    print(f"\nSaved {len(train_recs)} train and {len(test_recs)} test sequences to {OUT}")

if __name__ == "__main__":
    main()
