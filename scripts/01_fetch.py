#!/usr/bin/env python3
"""Phase 1: fetch complete genomes for the 9 viruses.
Uses datasets summary (metadata) + efetch (FASTA download).
Reports counts before downloading."""
import json, subprocess, time, csv
from pathlib import Path

OUT = Path("data/raw/ncbi")
OUT.mkdir(parents=True, exist_ok=True)

FETCH_PLAN = [
    ("dengue",           "Orthoflavivirus denguei",                  3000),
    ("west_nile",        "West Nile virus",                          1500),
    ("yellow_fever",     "Yellow fever virus",                       1000),
    ("ebola_zaire",      "Zaire ebolavirus",                         1000),
    ("ebola_sudan",      "Sudan ebolavirus",                          300),
    ("ebola_bundibugyo", "Bundibugyo ebolavirus",                      60),
    ("ebola_taiforest",  "Tai Forest ebolavirus",                      30),
    ("ebola_reston",     "Reston ebolavirus",                          50),
    ("marburg",          "Marburgvirus",                              300),
    ("rabies",           "Rabies lyssavirus",                        3000),
    ("rvfv",             "Rift Valley fever virus",                   800),
    ("cchf",             "Crimean-Congo hemorrhagic fever virus",    1000),
    ("chikungunya",      "Chikungunya virus",                        2000),
]


def fetch_summary(label, taxon, limit):
    jsonl = OUT / f"{label}.jsonl"
    if jsonl.exists() and jsonl.stat().st_size > 0:
        n = sum(1 for _ in open(jsonl))
        print(f"  summary cached ({n} records)")
        return n
    print(f"  probing {taxon} (limit {limit})")
    with open(jsonl, "w") as f:
        r = subprocess.run([
            "datasets", "summary", "virus", "genome", "taxon", taxon,
            "--complete-only", "--limit", str(limit), "--as-json-lines"
        ], stdout=f, stderr=subprocess.PIPE, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"  FAILED (rc={r.returncode})")
        print(f"  stderr: {r.stderr[:300]}")
        jsonl.unlink(missing_ok=True)
        return 0
    n = sum(1 for _ in open(jsonl))
    print(f"  {n} records available")
    return n


def download_fastas(label):
    jsonl = OUT / f"{label}.jsonl"
    fasta = OUT / f"{label}.fasta"
    if fasta.exists() and fasta.stat().st_size > 0:
        n = sum(1 for l in open(fasta) if l.startswith(">"))
        print(f"  fasta cached ({n} seqs)")
        return n
    accessions = []
    for line in open(jsonl):
        try:
            acc = json.loads(line).get("accession")
            if acc:
                accessions.append(acc)
        except Exception:
            pass
    if not accessions:
        print("  no accessions")
        return 0
    print(f"  downloading {len(accessions)} sequences via efetch")
    with open(fasta, "w") as f:
        for i in range(0, len(accessions), 100):
            batch = accessions[i:i+100]
            try:
                r = subprocess.run([
                    "efetch", "-db", "nuccore",
                    "-id", ",".join(batch), "-format", "fasta"
                ], capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    f.write(r.stdout)
            except subprocess.TimeoutExpired:
                print(f"    batch {i//100+1} TIMEOUT")
            time.sleep(0.4)
            if (i // 100 + 1) % 5 == 0:
                done = min(i + 100, len(accessions))
                print(f"    ... {done}/{len(accessions)}")
    n = sum(1 for l in open(fasta) if l.startswith(">"))
    print(f"  saved {n} sequences")
    return n


def main():
    report = []
    for label, taxon, limit in FETCH_PLAN:
        print(f"\n=== {label}: {taxon} ===")
        available = fetch_summary(label, taxon, limit)
        if available == 0:
            report.append({"label": label, "taxon": taxon,
                          "available": 0, "downloaded": 0, "status": "FAILED"})
            continue
        downloaded = download_fastas(label)
        status = "OK" if downloaded >= 30 else "THIN"
        report.append({"label": label, "taxon": taxon,
                      "available": available, "downloaded": downloaded,
                      "status": status})

    print("\n" + "=" * 70)
    print("FETCH SUMMARY")
    print("=" * 70)
    print(f"{'label':22s} {'available':>10s} {'downloaded':>12s} {'status':>10s}")
    for r in report:
        print(f"{r['label']:22s} {r['available']:>10d} {r['downloaded']:>12d} {r['status']:>10s}")

    with open(OUT / "fetch_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label","taxon","available","downloaded","status"])
        w.writeheader(); w.writerows(report)
    print(f"\nReport: {OUT}/fetch_report.csv")
    print(f"FASTAs: {OUT}/")


if __name__ == "__main__":
    main()
