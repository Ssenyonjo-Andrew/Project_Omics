"""Extract collection/release dates for all 783 sequences."""
import json, subprocess, tempfile, re
from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
from Bio import SeqIO

OUT_CSV = "data/processed/master_dated.csv"


def parse_date(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = datetime.strptime(s, fmt)
            return (d.year, d.month, d.day)
        except ValueError:
            pass
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        return (int(m.group(0)), 1, 1)
    return None


def extract_from_jsonl(path):
    out = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    for line in open(path):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        acc = rec.get("accession", "")
        if not acc:
            continue
        base = acc.split(".")[0]
        for src, getter in [
            ("collection_date", lambda r: (r.get("isolate") or {}).get("collection_date")),
            ("release_date",    lambda r: r.get("release_date")),
            ("update_date",     lambda r: r.get("update_date")),
        ]:
            val = getter(rec)
            if val:
                d = parse_date(val)
                if d:
                    out[base] = (d[0], d[1], d[2], src)
                    break
    return out


def extract_sarscov2_from_ncbi(accessions):
    """esummary-based -- much lighter than GenBank, no truncation."""
    out = {}
    BATCH = 100
    for i in range(0, len(accessions), BATCH):
        batch = accessions[i:i+BATCH]
        print(f"  esummary batch {i//BATCH+1}: {len(batch)} accessions")
        try:
            r = subprocess.run(
                ["esummary", "-db", "nuccore", "-id", ",".join(batch), "-format", "json"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0 or not r.stdout.strip():
                print(f"    FAILED")
                continue
            data = json.loads(r.stdout)
            result = data.get("result", {})
            for uid, rec in result.items():
                if not isinstance(rec, dict):
                    continue
                acc = rec.get("accessionversion") or rec.get("caption", "")
                if not acc:
                    continue
                base = acc.split(".")[0]
                cd = rec.get("createdate", "") or rec.get("updatedate", "")
                d = parse_date(cd)
                if d:
                    out[base] = (d[0], d[1], d[2], "ncbi_createdate")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            print(f"    error: {e}")
    return out




def main():
    df = pd.read_csv("data/processed/master_expanded.csv")
    df["slug"] = df["sequence_file_path"].apply(lambda p: Path(p).stem)
    print(f"Total: {len(df)} sequences across {df['slug'].nunique()} species\n")

    all_dates = {}
    for slug in sorted(df["slug"].unique()):
        if slug == "sars_cov_2":
            continue
        path = Path(f"data/raw/expanded/{slug}.jsonl")
        d = extract_from_jsonl(path)
        all_dates.update(d)
        src_counts = dict(Counter(v[3] for v in d.values()))
        print(f"  {slug:15s} {len(d):3d} dates  {src_counts}")

    sarscov2 = df[df["slug"] == "sars_cov_2"]["accession"].tolist()
    if sarscov2:
        print(f"\n  sars_cov_2: efetch for {len(sarscov2)} accessions")
        d = extract_sarscov2_from_ncbi(sarscov2)
        all_dates.update(d)
        src_counts = dict(Counter(v[3] for v in d.values()))
        print(f"  sars_cov_2      {len(d):3d} dates  {src_counts}")

    def get(acc, idx):
        base = acc.split(".")[0]
        return all_dates.get(base, (None, None, None, None))[idx]

    df["year"]         = df["accession"].map(lambda a: get(a, 0))
    df["month"]        = df["accession"].map(lambda a: get(a, 1))
    df["day"]          = df["accession"].map(lambda a: get(a, 2))
    df["date_source"]  = df["accession"].map(lambda a: get(a, 3))

    print(f"\n=== COVERAGE ===")
    print(f"With year: {df['year'].notna().sum()} / {len(df)}")
    print(f"\nDate source:")
    print(df["date_source"].value_counts(dropna=False).to_string())
    print(f"\nPer-species year range:")
    for slug in sorted(df["slug"].unique()):
        sub = df[(df["slug"] == slug) & df["year"].notna()]
        if len(sub) == 0:
            print(f"  {slug:15s} NO DATES")
            continue
        print(f"  {slug:15s} {len(sub):3d} dates, "
              f"median {int(sub['year'].median())}, "
              f"range {int(sub['year'].min())}-{int(sub['year'].max())}")

    df.drop(columns=["slug"]).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
