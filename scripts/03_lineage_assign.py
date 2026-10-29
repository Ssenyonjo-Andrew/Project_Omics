#!/usr/bin/env python3
"""Phase 3: assign lineage labels. Header parsing + clustering fallback."""
import re, subprocess, shutil, tempfile
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from Bio import SeqIO

IN_DIR  = Path("data/processed/filtered")
OUT_DIR = Path("data/processed/lineage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINEAGE_PATTERNS = {
    "dengue": [
        (r"dengue\s*(?:virus\s*)?(?:type\s*)?4\b|DENV[-\s]?4\b|DENV[-\s]?IV\b", "DENV-4"),
        (r"dengue\s*(?:virus\s*)?(?:type\s*)?3\b|DENV[-\s]?3\b|DENV[-\s]?III\b", "DENV-3"),
        (r"dengue\s*(?:virus\s*)?(?:type\s*)?2\b|DENV[-\s]?2\b|DENV[-\s]?II\b", "DENV-2"),
        (r"dengue\s*(?:virus\s*)?(?:type\s*)?1\b|DENV[-\s]?1\b|DENV[-\s]?I\b", "DENV-1"),
    ],
    "west_nile": [
        (r"lineage\s*(?:2|II)|WNV[-\s]?(?:L|lineage)[-\s]?2|kunjin", "lineage_2"),
        (r"lineage\s*(?:1|I)|WNV[-\s]?(?:L|lineage)[-\s]?1", "lineage_1"),
    ],
    "yellow_fever": [
        (r"east\s*african|YFV[-\s]?EA|uganda|ethiopia|kenya", "East_African"),
        (r"west\s*african|YFV[-\s]?WA", "West_African"),
        (r"angola|central\s*african|YFV[-\s]?CA", "Angola_CA"),
        (r"south\s*american|brazil|peru|bolivia|colombia", "South_American"),
    ],
    "rabies": [
        (r"africa[-\s]?2\b|AF2\b|africa\s*2", "Africa_2"),
        (r"africa[-\s]?1\b|AF1\b|africa\s*1", "Africa_1"),
        (r"arctic|ARC\b", "Arctic"),
        (r"asian|ASIA", "Asian"),
        (r"cosmopolitan|COSMO", "Cosmopolitan"),
    ],
    "chikungunya": [
        (r"east[/,\s]*central[/,\s]*south\s*afric|ECSA", "ECSA"),
        (r"west\s*african|WAF", "West_African"),
        (r"asian|ASIA", "Asian"),
    ],
}


def parse_lineage(virus, description):
    desc = description.lower()
    for pattern, lineage in LINEAGE_PATTERNS.get(virus, []):
        if re.search(pattern, desc, re.IGNORECASE):
            return lineage
    return None


def cluster_fill_labels(virus, fasta):
    """Parse headers, then cluster to fill unparsed sequences from neighbours."""
    print(f"\n=== {virus} ===")
    records = {rec.id: rec for rec in SeqIO.parse(fasta, "fasta")}
    print(f"  {len(records)} sequences")

    labels = {}
    for rid, rec in records.items():
        lin = parse_lineage(virus, rec.description)
        if lin:
            labels[rid] = lin

    print(f"  header-parsed: {len(labels)}/{len(records)} "
          f"({100*len(labels)/len(records):.1f}%)")
    print(f"  lineage counts (parsed only): {dict(Counter(labels.values()))}")

    unparsed = [rid for rid in records if rid not in labels]

    if unparsed:
        print(f"  clustering to infer {len(unparsed)} unparsed sequences...")
        tmp = Path(tempfile.mkdtemp())
        try:
            out = tmp / "c"
            subprocess.run([
                "mmseqs", "easy-cluster", str(fasta), str(out), str(tmp / "t"),
                "--min-seq-id", "0.95", "-c", "0.8",
                "--threads", "4",
            ], check=True, capture_output=True, text=True, timeout=3600)

            cluster_file = Path(str(out) + "_cluster.tsv")
            clusters = defaultdict(list)
            for line in open(cluster_file):
                rep, member = line.strip().split("\t")
                clusters[rep].append(member)

            for rep, members in clusters.items():
                labeled_here = [labels[m] for m in members if m in labels]
                if not labeled_here:
                    continue
                majority = Counter(labeled_here).most_common(1)[0][0]
                for m in members:
                    if m not in labels:
                        labels[m] = majority
        except subprocess.CalledProcessError as e:
            print(f"  MMseqs2 failed: {e.stderr[:200]}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        still = [rid for rid in records if rid not in labels]
        print(f"  after clustering: {len(labels)}/{len(records)} labelled")
        print(f"  still unparsed: {len(still)}")

    rows = [{
        "accession": rid,
        "sequence_file": str(fasta),
        "virus": virus,
        "lineage": labels.get(rid, "unknown"),
        "length": len(records[rid].seq),
    } for rid in records]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{virus}_lineages.csv", index=False)
    print(f"  final: {dict(df['lineage'].value_counts())}")
    return df


def process_ebola():
    print(f"\n=== ebola ===")
    species_map = {
        "ebola_zaire":      "Zaire",
        "ebola_sudan":      "Sudan",
        "ebola_bundibugyo": "Bundibugyo",
        "ebola_taiforest":  "Tai_Forest",
        "ebola_reston":     "Reston",
    }
    rows = []
    for source, species in species_map.items():
        fasta = IN_DIR / f"{source}.fasta"
        if not fasta.exists():
            continue
        for rec in SeqIO.parse(fasta, "fasta"):
            lineage = "Zaire" if species == "Zaire" else "Other_Ebolavirus"
            rows.append({
                "accession": rec.id,
                "sequence_file": str(fasta),
                "virus": "ebola",
                "species": species,
                "lineage": lineage,
                "length": len(rec.seq),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ebola_lineages.csv", index=False)
    print(f"  {len(df)} sequences")
    print(f"  Zaire (test): {(df['lineage']=='Zaire').sum()}")
    print(f"  Other (train): {(df['lineage']=='Other_Ebolavirus').sum()}")
    print(f"  per species: {dict(df['species'].value_counts())}")
    return df


def main():
    all_dfs = []
    for v in ["dengue", "west_nile", "yellow_fever", "rabies", "chikungunya"]:
        fasta = IN_DIR / f"{v}.fasta"
        if fasta.exists():
            all_dfs.append(cluster_fill_labels(v, fasta))

    all_dfs.append(process_ebola())

    master = pd.concat(all_dfs, ignore_index=True)
    master.to_csv(OUT_DIR / "master_lineage.csv", index=False)

    print("\n" + "=" * 72)
    print("FINAL LINEAGE COUNTS")
    print("=" * 72)
    print(pd.crosstab(master["virus"], master["lineage"]).to_string())

    print("\n" + "=" * 72)
    print("WARNINGS")
    print("=" * 72)
    for virus in master["virus"].unique():
        sub = master[master["virus"] == virus]
        unk = (sub["lineage"] == "unknown").sum()
        if unk:
            print(f"  {virus}: {unk} unparsed")

    for (virus, lineage), count in master.groupby(["virus","lineage"]).size().items():
        if lineage != "unknown" and count < 30:
            print(f"  THIN: {virus}/{lineage} = {count}")

    print(f"\nWrote {OUT_DIR}/master_lineage.csv ({len(master)} sequences)")


if __name__ == "__main__":
    main()
