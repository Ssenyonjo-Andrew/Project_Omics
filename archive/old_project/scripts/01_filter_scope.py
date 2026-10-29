import subprocess
import time
from pathlib import Path
import pandas as pd

TAXONOMY_PATH = "data/raw/ictv_output/taxonomy_table.csv"
FASTA_OUT     = "data/processed/sequences.fasta"
LABELS_OUT    = "data/processed/labels.csv"
FAILED_OUT    = "data/processed/failed_accessions.txt"
BATCH_SIZE    = 100

# ICTV MSL41 species name -> short public name
TARGET_SPECIES = {
    "Betacoronavirus pandemicum":               "SARS-CoV-2",
    "Betacoronavirus cameli":                   "MERS-CoV",
    "Orthoebolavirus zairense":                 "Ebola virus",
    "Orthomarburgvirus marburgense":            "Marburg virus",
    "Mammarenavirus lassaense":                 "Lassa virus",
    "Henipavirus nipahense":                    "Nipah virus",
    "Henipavirus hendraense":                   "Hendra virus",
    "Orthoflavivirus zikaense":                 "Zika virus",
    "Orthoflavivirus denguei":                  "Dengue virus",
    "Orthoflavivirus flavi":                    "Yellow fever virus",
}

def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TAXONOMY_PATH)
    df_filtered = df[df["Species"].isin(TARGET_SPECIES)].copy()
    print(f"Matched {len(df_filtered)} VMR rows across "
          f"{df_filtered['Species'].nunique()} species")

    missing = set(TARGET_SPECIES) - set(df_filtered["Species"])
    if missing:
        print("WARNING - no rows in MSL41 for:")
        for m in missing:
            print(f"   {m}")

    df_filtered["accession"] = (
        df_filtered["Virus GENBANK accession"]
        .astype(str)
        .str.split(r"[;,]", regex=True)
        .str[0]
        .str.strip()
    )
    df_filtered = df_filtered[df_filtered["accession"].notna()]
    df_filtered = df_filtered[df_filtered["accession"] != "nan"]
    df_filtered = df_filtered.drop_duplicates(subset="accession", keep="first")
    print(f"Unique accessions: {len(df_filtered)}")

    labels = df_filtered[["accession", "Species", "Family", "Genus"]].copy()
    labels["virus_name"] = labels["Species"].map(TARGET_SPECIES)
    labels = labels.rename(columns={"Species": "species",
                                    "Family":  "family",
                                    "Genus":   "genus"})
    labels = labels[["accession", "virus_name", "family", "genus", "species"]]

    accessions = df_filtered["accession"].tolist()
    n_batches  = (len(accessions) + BATCH_SIZE - 1) // BATCH_SIZE
    fasta_chunks, failed = [], []

    for i in range(0, len(accessions), BATCH_SIZE):
        batch = accessions[i:i + BATCH_SIZE]
        idx   = i // BATCH_SIZE + 1
        print(f"Fetching batch {idx}/{n_batches} ({len(batch)} accessions)...")
        try:
            result = subprocess.run(
                ["efetch", "-db", "nucleotide",
                 "-id", ",".join(batch), "-format", "fasta"],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            failed.extend(batch)
            print("  timeout")
            continue

        if result.returncode != 0 or not result.stdout.strip():
            failed.extend(batch)
            print(f"  efetch failed (rc={result.returncode}): "
                  f"{result.stderr[:200]}")
            continue

        fasta_chunks.append(result.stdout)
        returned = {ln[1:].split()[0]
                    for ln in result.stdout.splitlines() if ln.startswith(">")}
        for acc in batch:
            if not any(r == acc or r.startswith(acc + ".") for r in returned):
                failed.append(acc)

        time.sleep(0.4)

    Path(FASTA_OUT).write_text("".join(fasta_chunks))
    labels = labels[~labels["accession"].isin(failed)]
    labels.to_csv(LABELS_OUT, index=False)
    Path(FAILED_OUT).write_text("\n".join(failed))

    print("\nDONE")
    print(f"  Sequences written : {len(labels)}")
    print(f"  Accessions failed : {len(failed)}")
    print(f"  FASTA            : {FASTA_OUT}")
    print(f"  Labels           : {LABELS_OUT}")
    print(f"  Failures logged  : {FAILED_OUT}")

if __name__ == "__main__":
    main()