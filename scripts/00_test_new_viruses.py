#!/usr/bin/env python3
"""Test if WNV, JEV, and YFV have parseable lineage metadata."""
import json, subprocess, re
from collections import Counter

TESTS = [
    ("west_nile", "West Nile virus", [(r"lineage\s*2", "Lineage_2"), (r"lineage\s*1", "Lineage_1")]),
    ("jev", "Japanese encephalitis virus", [(r"genotype\s*I\b", "GI"), (r"genotype\s*III", "GIII"), (r"genotype\s*V\b", "GV")]),
    ("yfv", "Yellow fever virus", [(r"west\s*african", "West_African"), (r"east\s*african", "East_African")])
]

for slug, taxon, patterns in TESTS:
    print(f"\n=== Testing {slug} ({taxon}) ===")
    # Get just 200 records to test
    r = subprocess.run([
        "datasets", "summary", "virus", "genome", "taxon", taxon,
        "--complete-only", "--limit", "200", "--as-json-lines"
    ], capture_output=True, text=True)
    
    counts = Counter()
    for line in r.stdout.splitlines():
        if not line.strip(): continue
        rec = json.loads(line)
        
        text = ""
        iso = rec.get("isolate", {}) or {}
        text += " " + str(iso.get("name", ""))
        text += " " + str(rec.get("description", ""))
        for lin in (rec.get("virus", {}) or {}).get("lineage", []):
            text += " " + str(lin.get("name", ""))
            
        matched = "unknown"
        for pattern, label in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched = label
                break
        counts[matched] += 1
        
    print(f"  Total records tested: {sum(counts.values())}")
    for k, v in counts.most_common():
        print(f"    {k}: {v}")
