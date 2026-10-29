#!/usr/bin/env python3
"""Assign lineage labels for 7-virus set: Dengue, Chikungunya, HCV, Norovirus, HBV, HIV-1, Mpox."""
import json, re, subprocess, tempfile, shutil
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from Bio import SeqIO

IN_FASTA = Path("data/raw/ncbi")
IN_JSONL = Path("data/raw/ncbi")
OUT_DIR = Path("data/processed/lineage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_jsonl(label):
    records = {}
    for line in open(IN_JSONL / f"{label}.jsonl"):
        try:
            rec = json.loads(line)
            acc = rec.get("accession")
            if acc:
                records[acc.split(".")[0]] = rec
        except Exception:
            pass
    return records

def extract_lineage_name(virus_rec, prefix):
    """Return the lineage name starting with prefix, e.g. 'Hepatitis C virus genotype'."""
    for lin in virus_rec.get("lineage", []):
        name = lin.get("name", "")
        if name.startswith(prefix):
            return name
    return None

def assign_hcv():
    recs = load_jsonl("hcv")
    fasta = IN_FASTA / "hcv.fasta"
    rows = []
    for seq in SeqIO.parse(fasta, "fasta"):
        base = seq.id.split(".")[0]
        rec = recs.get(base, {})
        lin = extract_lineage_name(rec.get("virus", {}), "Hepatitis C virus genotype")
        genotype = None
        if lin:
            m = re.search(r"genotype\s*(\d+)", lin)
            if m:
                genotype = m.group(1)
        rows.append({"accession": seq.id, "virus": "hcv",
                     "lineage": f"GT{genotype}" if genotype else "unknown",
                     "length": len(seq.seq), "fasta": str(fasta)})
    return pd.DataFrame(rows)

def assign_norovirus():
    recs = load_jsonl("norovirus")
    fasta = IN_FASTA / "norovirus.fasta"
    rows = []
    for seq in SeqIO.parse(fasta, "fasta"):
        base = seq.id.split(".")[0]
        rec = recs.get(base, {})
        lin = extract_lineage_name(rec.get("virus", {}), "Norovirus G")
        genogroup = None
        if lin:
            m = re.search(r"Norovirus\s+(G[IVX]+)", lin)
            if m:
                genogroup = m.group(1)
        rows.append({"accession": seq.id, "virus": "norovirus",
                     "lineage": genogroup or "unknown",
                     "length": len(seq.seq), "fasta": str(fasta)})
    return pd.DataFrame(rows)

def assign_hbv():
    recs = load_jsonl("hbv")
    fasta = IN_FASTA / "hbv.fasta"
    rows = []
    for seq in SeqIO.parse(fasta, "fasta"):
        base = seq.id.split(".")[0]
        rec = recs.get(base, {})
        iso = (rec.get("isolate") or {}).get("name", "")
        genotype = None
        m = re.search(r"HBV[-_]?([A-J])", iso, re.IGNORECASE)
        if m:
            genotype = m.group(1).upper()
        rows.append({"accession": seq.id, "virus": "hbv",
                     "lineage": f"GT{genotype}" if genotype else "unknown",
                     "length": len(seq.seq), "fasta": str(fasta)})
    return pd.DataFrame(rows)

# ---------- HIV-1 via PyHIV ----------
def assign_hiv1():
    fasta = IN_FASTA / "hiv1.fasta"
    out_dir = Path(tempfile.mkdtemp())
    try:
        r = subprocess.run(
            ["pyhiv", "run", str(fasta), "-o", str(out_dir), "-j", "4"],
            capture_output=True, text=True, timeout=7200
        )
        if r.returncode != 0:
            print(f"  PyHIV failed: {r.stderr[:400]}")
            return pd.DataFrame([{"accession": s.id, "virus": "hiv1",
                                  "lineage": "unknown", "length": len(s.seq),
                                  "fasta": str(fasta)} for s in SeqIO.parse(fasta, "fasta")])
        table = out_dir / "final_table.tsv"
        subtype_map = {}
        if table.exists():
            df = pd.read_csv(table, sep="\t")
            for _, row in df.iterrows():
                acc = str(row.get("sequence_name", "")).split(".")[0]
                subtype = str(row.get("best_subtype", "unknown"))
                subtype_map[acc] = subtype
        rows = []
        for seq in SeqIO.parse(fasta, "fasta"):
            base = seq.id.split(".")[0]
            rows.append({"accession": seq.id, "virus": "hiv1",
                         "lineage": subtype_map.get(base, "unknown"),
                         "length": len(seq.seq), "fasta": str(fasta)})
        return pd.DataFrame(rows)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

# ---------- Mpox via SNP positions ----------
# Reference NC_063383.1, positions are 1-based on the reference genome.
MPOX_SNPS = {
    "clade_I_marker":  {"pos": 3143, "ref": "C", "alt": "T"},  # C3143T -> clade I
    "clade_Ib_SNP1":  {"pos": 2967, "ref": "G", "alt": "A"},  # G2967A -> Ib
    "clade_Ib_SNP2":  {"pos": 3145, "ref": "G", "alt": "A"},  # G3145A -> Ib
    "clade_IIa_SNP":  {"pos": 2959, "ref": "C", "alt": "T"},  # C2959T -> IIa
    "non_IIb_SNP":    {"pos": 3099, "ref": "T", "alt": "G"},  # T3099G -> non-IIb
}
MPOX_REF_ACC = "NC_063383.1"

def fetch_mpox_ref():
    ref_path = OUT_DIR / "mpox_ref.fasta"
    if ref_path.exists() and ref_path.stat().st_size > 0:
        return ref_path
    r = subprocess.run(
        ["efetch", "-db", "nuccore", "-id", MPOX_REF_ACC, "-format", "fasta"],
        capture_output=True, text=True, timeout=120
    )
    ref_path.write_text(r.stdout)
    return ref_path

def assign_mpox():
    fasta = IN_FASTA / "mpox.fasta"
    ref_rec = next(SeqIO.parse(fetch_mpox_ref(), "fasta"))
    ref = str(ref_rec.seq).upper()
    ref_map = {MPOX_SNPS[k]["pos"]: MPOX_SNPS[k] for k in MPOX_SNPS}

    rows = []
    for seq in SeqIO.parse(fasta, "fasta"):
        s = str(seq.seq).upper()
        clade = "unknown"
        # align-free: assume lengths are near reference; pad/truncate to reference length
        # For positions up to ~3200, most complete genomes align without indels.
        # Take a simple direct slice if lengths are close.
        if abs(len(s) - len(ref)) > 10000:
            # too diverged from reference for direct coordinate lookup
            rows.append({"accession": seq.id, "virus": "mpox",
                         "lineage": "unknown", "length": len(seq.seq),
                         "fasta": str(fasta)})
            continue
        # try direct position lookup
        c3143 = s[3142] if len(s) > 3143 else "?"
        g2967 = s[2966] if len(s) > 2967 else "?"
        g3145 = s[3144] if len(s) > 3145 else "?"
        c2959 = s[2958] if len(s) > 2959 else "?"
        t3099 = s[3098] if len(s) > 3099 else "?"

        is_clade_I = (c3143 == "T")  # C3143T
        is_Ib = is_clade_I and g2967 == "A" and g3145 == "A"
        is_IIa = (c3143 == "C") and c2959 == "T"
        is_IIb = (c3143 == "C") and t3099 == "T"

        if is_Ib:
            clade = "Ib"
        elif is_clade_I:
            clade = "Ia"
        elif is_IIb:
            clade = "IIb"
        elif is_IIa:
            clade = "IIa"
        elif c3143 == "C":
            clade = "II_other"

        rows.append({"accession": seq.id, "virus": "mpox",
                     "lineage": clade, "length": len(seq.seq),
                     "fasta": str(fasta)})
    return pd.DataFrame(rows)


def main():
    dfs = []

    print("Dengue: reusing existing CSV")
    dfs.append(pd.read_csv(OUT_DIR / "dengue_lineages.csv"))

    print("Chikungunya: reusing existing CSV")
    dfs.append(pd.read_csv(OUT_DIR / "chikungunya_lineages.csv"))

    print("\nHCV: parsing genotype from virus.lineage")
    hcv = assign_hcv()
    hcv.to_csv(OUT_DIR / "hcv_lineages.csv", index=False)
    dfs.append(hcv)
    print(f"  {dict(hcv['lineage'].value_counts())}")

    print("\nNorovirus: parsing genogroup from virus.lineage")
    noro = assign_norovirus()
    noro.to_csv(OUT_DIR / "norovirus_lineages.csv", index=False)
    dfs.append(noro)
    print(f"  {dict(noro['lineage'].value_counts())}")

    print("\nHBV: regex on isolate.name")
    hbv = assign_hbv()
    hbv.to_csv(OUT_DIR / "hbv_lineages.csv", index=False)
    dfs.append(hbv)
    print(f"  {dict(hbv['lineage'].value_counts())}")

    print("\nHIV-1: PyHIV subtyping (may take 20-40 min)")
    hiv = assign_hiv1()
    hiv.to_csv(OUT_DIR / "hiv1_lineages.csv", index=False)
    dfs.append(hiv)
    print(f"  {dict(hiv['lineage'].value_counts())}")

    print("\nMpox: SNP-based clade assignment")
    mpox = assign_mpox()
    mpox.to_csv(OUT_DIR / "mpox_lineages.csv", index=False)
    dfs.append(mpox)
    print(f"  {dict(mpox['lineage'].value_counts())}")

    master = pd.concat(dfs, ignore_index=True)
    master.to_csv(OUT_DIR / "master_lineage_7v.csv", index=False)

    print("\n" + "=" * 72)
    print("FINAL LINEAGE COUNTS (7-virus set)")
    print("=" * 72)
    print(pd.crosstab(master["virus"], master["lineage"]).to_string())

    print("\n" + "=" * 72)
    print("TEST HOLDOUTS")
    print("=" * 72)
    test_holdouts = {
        "dengue": "DENV-4",
        "chikungunya": "Asian",
        "hcv": "GT4",
        "norovirus": "GIV",
        "hbv": "GTH",
        "hiv1": "C",
        "mpox": "Ia",  # or "Ib" depending on what the SNP calls show
    }
    for v, holdout in test_holdouts.items():
        sub = master[master["virus"] == v]
        n_test = (sub["lineage"] == holdout).sum()
        n_train = len(sub) - n_test
        print(f"  {v:14s} holdout={holdout:8s}  test={n_test:5d}  train={n_train:5d}")

    print(f"\nWrote {OUT_DIR}/master_lineage_7v.csv ({len(master)} sequences)")


if __name__ == "__main__":
    main()
