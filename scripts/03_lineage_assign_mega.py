#!/usr/bin/env python3
"""Assign lineages for the 12-virus pool and report holdout viability."""
import json, re
from pathlib import Path
from collections import Counter
import pandas as pd

RAW = Path("data/raw/ncbi")
OUT = Path("data/processed/lineage")
OUT.mkdir(parents=True, exist_ok=True)

# We only need to define the regex patterns to find the lineage in the metadata
PATTERNS = {
    "dengue":      [(r"type\s*4|DENV-?4", "DENV-4"), (r"type\s*1|DENV-?1", "DENV-1")],
    "chikungunya": [(r"asian", "Asian"), (r"ecsa|east", "ECSA")],
    "norovirus":   [(r"GII", "GII"), (r"GI", "GI")],
    "measles":     [(r"B3", "B3"), (r"D8", "D8"), (r"D4", "D4")],
    "rabies":      [(r"africa-?2|AF2", "Africa_2"), (r"africa-?1|AF1", "Africa_1"), (r"cosmo", "Cosmopolitan")],
    "ebola":       [(r"zaire", "Zaire"), (r"sudan", "Sudan")],
    "zika":        [(r"asian", "Asian"), (r"african", "African")],
    "mpox":        [(r"clade\s*(IIb|2b)", "IIb"), (r"clade\s*(IIa|2a)", "IIa"), (r"clade\s*(I|1)\b", "I")],
    "hsv1":        [(r"clade\s*2", "Clade_2"), (r"clade\s*1", "Clade_1")],
    "adeno_c":     [(r"type\s*6", "Type_6"), (r"type\s*5", "Type_5"), (r"type\s*2", "Type_2")],
    "hpv16":       [(r"lineage\s*d", "Lineage_D"), (r"lineage\s*a", "Lineage_A")],
    "hbv":         [(r"genotype\s*H|HBV-?H", "GT_H"), (r"genotype\s*B|HBV-?B", "GT_B")],
}

def get_text(rec):
    """Combine all possible text fields where a lineage might hide."""
    text = ""
    iso = rec.get("isolate", {}) or {}
    text += " " + str(iso.get("name", ""))
    text += " " + str(rec.get("description", ""))
    virus = rec.get("virus", {}) or {}
    for lin in virus.get("lineage", []):
        text += " " + str(lin.get("name", ""))
    return text.lower()

def process(virus):
    jsonl = RAW / f"{virus}.jsonl"
    if not jsonl.exists(): return {}
    
    counts = Counter()
    total = 0
    for line in open(jsonl):
        try:
            rec = json.loads(line)
            if not rec.get("accession"): continue
            total += 1
            text = get_text(rec)
            
            matched = "unknown"
            for pattern, label in PATTERNS.get(virus, []):
                if re.search(pattern, text, re.IGNORECASE):
                    matched = label
                    break
            counts[matched] += 1
        except Exception:
            pass
            
    return total, counts

def main():
    print(f"{'Virus':15s} {'Total':>6s} | Top Lineages Found (Candidate Holdouts)")
    print("-" * 75)
    for v in PATTERNS.keys():
        res = process(v)
        if not res:
            print(f"{v:15s} MISSING METADATA")
            continue
        total, counts = res
        top = [f"{k}: {n}" for k, n in counts.most_common(4) if k != "unknown"]
        unknown = counts.get("unknown", 0)
        print(f"{v:15s} {total:6d} | {', '.join(top)} (Unknown: {unknown})")

if __name__ == "__main__":
    main()
