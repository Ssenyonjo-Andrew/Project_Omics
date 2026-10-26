import json
import subprocess
import time
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/raw/expanded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# slug -> (candidate NCBI taxon names, virus_name, family, genus, ictv species)
SPECIES_CONFIG = {
    "dengue": (
        ["Orthoflavivirus denguei", "Dengue virus"],
        "Dengue virus", "Flaviviridae", "Orthoflavivirus", "Orthoflavivirus denguei"),
    "ebola": (
        ["Orthoebolavirus zairense", "Zaire ebolavirus"],
        "Ebola virus", "Filoviridae", "Orthoebolavirus", "Orthoebolavirus zairense"),
    "mers": (
        ["Betacoronavirus cameli", "Middle East respiratory syndrome coronavirus"],
        "MERS-CoV", "Coronaviridae", "Betacoronavirus", "Betacoronavirus cameli"),
    "yellow_fever": (
        ["Orthoflavivirus flavi", "Yellow fever virus"],
        "Yellow fever virus", "Flaviviridae", "Orthoflavivirus", "Orthoflavivirus flavi"),
    "sars_cov_2": (
        ["Betacoronavirus pandemicum", "SARS-CoV-2", "Severe acute respiratory syndrome coronavirus 2"],
        "SARS-CoV-2", "Coronaviridae", "Betacoronavirus", "Betacoronavirus pandemicum"),
    "zika": (
        ["Orthoflavivirus zikaense", "Zika virus"],
        "Zika virus", "Flaviviridae", "Orthoflavivirus", "Orthoflavivirus zikaense"),
    "marburg": (
        ["Orthomarburgvirus marburgense", "Marburgvirus", "Marburg virus"],
        "Marburg virus", "Filoviridae", "Orthomarburgvirus", "Orthomarburgvirus marburgense"),
    "lassa": (
        ["Mammarenavirus lassaense", "Lassa virus"],
        "Lassa virus", "Arenaviridae", "Mammarenavirus", "Mammarenavirus lassaense"),
    "nipah": (
        ["Henipavirus nipahense", "Nipah virus"],
        "Nipah virus", "Paramyxoviridae", "Henipavirus", "Henipavirus nipahense"),
    "hendra": (
        ["Henipavirus hendraense", "Hendra virus"],
        "Hendra virus", "Paramyxoviridae", "Henipavirus", "Henipavirus hendraense"),
}

LIMIT_PER_SPECIES = 150
BATCH_SIZE        = 100
SLEEP_BETWEEN     = 0.4

def try_summary(slug, candidates):
    """Try each candidate taxon name. Return the jsonl path that worked, or None."""
    jsonl = OUT_DIR / f"{slug}.jsonl"
    if jsonl.exists() and jsonl.stat().st_size > 0:
        print(f"  [{slug}] summary cached ({jsonl.stat().st_size} bytes)")
        return jsonl

    for taxon in candidates:
        print(f"  [{slug}] trying taxon: {taxon!r}")
        try:
            with open(jsonl, "w") as f:
                subprocess.run(
                    ["datasets", "summary", "virus", "genome", "taxon", taxon,
                     "--complete-only", "--limit", str(LIMIT_PER_SPECIES),
                     "--as-json-lines"],
                    stdout=f, stderr=subprocess.DEVNULL,
                    check=True, timeout=180,
                )
            if jsonl.stat().st_size > 0:
                print(f"  [{slug}] OK with {taxon!r} ({jsonl.stat().st_size} bytes)")
                return jsonl
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  [{slug}] FAILED with {taxon!r}")
            jsonl.unlink(missing_ok=True)
    print(f"  [{slug}] ALL TAXON NAMES FAILED - skipping")
    return None

def fetch_fasta(slug, jsonl):
    accessions = []
    for line in open(jsonl):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        acc = rec.get("accession")
        if acc:
            accessions.append(acc)

    if not accessions:
        print(f"  [{slug}] NO ACCESSIONS in summary")
        return 0

    fasta = OUT_DIR / f"{slug}.fasta"
    print(f"  [{slug}] efetch {len(accessions)} accessions")
    with open(fasta, "w") as fout:
        for i in range(0, len(accessions), BATCH_SIZE):
            batch = accessions[i:i+BATCH_SIZE]
            try:
                res = subprocess.run(
                    ["efetch", "-db", "nucleotide",
                     "-id", ",".join(batch), "-format", "fasta"],
                    capture_output=True, text=True, timeout=180,
                )
                if res.returncode == 0:
                    fout.write(res.stdout)
            except subprocess.TimeoutExpired:
                print(f"  [{slug}] batch timeout")
            time.sleep(SLEEP_BETWEEN)
    return len(accessions)

def build_master():
    rows = []
    for slug, (_, vname, family, genus, species) in SPECIES_CONFIG.items():
        jsonl = OUT_DIR / f"{slug}.jsonl"
        fasta = OUT_DIR / f"{slug}.fasta"
        if not jsonl.exists() or not fasta.exists():
            print(f"  [{slug}] MISSING for master - skipped")
            continue

        in_fasta = set()
        for line in open(fasta):
            if line.startswith(">"):
                in_fasta.add(line[1:].split()[0].split(".")[0])

        for line in open(jsonl):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc = rec.get("accession", "")
            if acc.split(".")[0] not in in_fasta:
                continue
            rows.append({
                "accession": acc,
                "sequence_file_path": str(fasta),
                "virus_name": vname,
                "family": family,
                "genus": genus,
                "species": species,
                "cluster_id": pd.NA,
                "split": pd.NA,
            })

    df = pd.DataFrame(rows).drop_duplicates("accession")
    df = df[["accession", "sequence_file_path", "virus_name",
             "family", "genus", "species", "cluster_id", "split"]]
    out = "data/processed/master_expanded.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")
    if len(df):
        print(df["virus_name"].value_counts().to_string())

def main():
    print(f"Expanding {len(SPECIES_CONFIG)} species, limit {LIMIT_PER_SPECIES} each\n")
    for slug, (candidates, *_rest) in SPECIES_CONFIG.items():
        jsonl = try_summary(slug, candidates)
        if jsonl is not None:
            fetch_fasta(slug, jsonl)
    build_master()

if __name__ == "__main__":
    main()