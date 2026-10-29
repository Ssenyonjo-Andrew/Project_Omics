#!/usr/bin/env python3
"""Phase 2: filter to complete genomes (unsegmented) or M segment (segmented)."""
from pathlib import Path
from Bio import SeqIO

IN_DIR  = Path("data/raw/ncbi")
OUT_DIR = Path("data/processed/filtered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_LENGTHS = {
    "dengue":           10663,
    "west_nile":        11030,
    "yellow_fever":     10862,
    "ebola_zaire":      18921,
    "ebola_sudan":      18875,
    "ebola_bundibugyo": 18805,
    "ebola_taiforest":  18935,
    "ebola_reston":     18887,
    "marburg":          19114,
    "rabies":           11923,
    "rvfv":              3885,   # M segment
    "cchf":              5368,   # M segment
    "chikungunya":      11764,
}

TOLERANCE = 0.20

def filter_one(label):
    ref = REFERENCE_LENGTHS[label]
    lo  = int(ref * (1 - TOLERANCE))
    hi  = int(ref * (1 + TOLERANCE))
    src = IN_DIR / f"{label}.fasta"
    out = OUT_DIR / f"{label}.fasta"
    if not src.exists():
        return {"label": label, "total": 0, "kept": 0, "rejected": 0,
                "lo": lo, "hi": hi, "status": "MISSING"}
    kept, rejected = [], 0
    for rec in SeqIO.parse(src, "fasta"):
        if lo <= len(rec.seq) <= hi:
            kept.append(rec)
        else:
            rejected += 1
    SeqIO.write(kept, out, "fasta")
    return {"label": label, "total": len(kept) + rejected, "kept": len(kept),
            "rejected": rejected, "lo": lo, "hi": hi,
            "status": "OK" if len(kept) >= 30 else "THIN"}

def main():
    rows = [filter_one(l) for l in sorted(REFERENCE_LENGTHS)]
    print(f"{'virus':22s} {'total':>6s} {'kept':>6s} {'rej':>5s} "
          f"{'range':>14s} {'status':>7s}")
    print("-" * 72)
    for r in rows:
        print(f"{r['label']:22s} {r['total']:>6d} {r['kept']:>6d} "
              f"{r['rejected']:>5d} {f'{r[chr(108)+chr(111)]}-{r[chr(104)+chr(105)]}':>14s} "
              f"{r['status']:>7s}")
    total = sum(r["kept"] for r in rows)
    print(f"\nTotal kept: {total}")
    print(f"Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
