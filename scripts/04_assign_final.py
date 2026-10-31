#!/usr/bin/env python3
import json, re, subprocess
from pathlib import Path
from collections import Counter
import pandas as pd
from Bio import SeqIO

RAW = Path("data/raw/ncbi")
OUT = Path("data/processed/lineage")
OUT.mkdir(parents=True, exist_ok=True)

def parse_json(virus, patterns):
    jsonl = RAW / f"{virus}.jsonl"
    if not jsonl.exists(): return {}
    out = {}
    for line in open(jsonl):
        try:
            rec = json.loads(line)
            acc = rec.get("accession")
            if not acc: continue
            
            text = ""
            iso = rec.get("isolate", {}) or {}
            text += " " + str(iso.get("name", ""))
            text += " " + str(rec.get("description", ""))
            for lin in (rec.get("virus", {}) or {}).get("lineage", []):
                text += " " + str(lin.get("name", ""))
            
            for pattern, label in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    out[acc.split(".")[0]] = label
                    break
        except Exception:
            pass
    return out

def parse_chikungunya():
    fasta = RAW / "chikungunya.fasta"
    out = {}
    for rec in SeqIO.parse(fasta, "fasta"):
        base = rec.id.split(".")[0]
        desc = rec.description.lower()
        if "asian" in desc or "asia" in desc: out[base] = "Asian"
        elif "ecsa" in desc or "east" in desc or "africa" in desc: out[base] = "ECSA"
    return out

def parse_adeno():
    fasta = RAW / "adeno_c.fasta"
    out = {}
    for rec in SeqIO.parse(fasta, "fasta"):
        base = rec.id.split(".")[0]
        desc = rec.description.lower()
        m = re.search(r"adenovirus\s*(\d+)|HAdV-C(\d+)", desc)
        if m:
            st = m.group(1) or m.group(2)
            out[base] = f"Type_{st}"
    return out

def parse_mpox():
    fasta = RAW / "mpox.fasta"
    ref_path = OUT / "mpox_ref.fasta"
    if not ref_path.exists():
        r = subprocess.run(["efetch", "-db", "nuccore", "-id", "NC_063383.1", "-format", "fasta"], capture_output=True, text=True)
        ref_path.write_text(r.stdout)
    ref = str(next(SeqIO.parse(ref_path, "fasta")).seq).upper()
    
    out = {}
    for rec in SeqIO.parse(fasta, "fasta"):
        base = rec.id.split(".")[0]
        s = str(rec.seq).upper()
        if abs(len(s) - len(ref)) > 10000: continue
        
        c3143 = s[3142] if len(s) > 3143 else "?"
        g2967 = s[2966] if len(s) > 2967 else "?"
        g3145 = s[3144] if len(s) > 3145 else "?"
        c2959 = s[2958] if len(s) > 2959 else "?"
        t3099 = s[3098] if len(s) > 3099 else "?"

        is_clade_I = (c3143 == "T")
        is_Ib = is_clade_I and g2967 == "A" and g3145 == "A"
        is_IIa = (c3143 == "C") and c2959 == "T"
        is_IIb = (c3143 == "C") and t3099 == "T"

        if is_Ib: out[base] = "Ib"
        elif is_clade_I: out[base] = "Ia"
        elif is_IIb: out[base] = "IIb"
        elif is_IIa: out[base] = "IIa"
        elif c3143 == "C": out[base] = "II_other"
    return out

def main():
    assignments = {}
    print("Parsing metadata...")
    assignments["dengue"] = parse_json("dengue", [(r"type\s*4|DENV-?4", "DENV-4"), (r"type\s*1|DENV-?1", "DENV-1")])
    assignments["norovirus"] = parse_json("norovirus", [(r"GII", "GII"), (r"GI", "GI")])
    assignments["measles"] = parse_json("measles", [(r"B3", "B3"), (r"D8", "D8")])
    
    print("Parsing headers...")
    assignments["chikungunya"] = parse_chikungunya()
    assignments["adeno_c"] = parse_adeno()
    
    print("Parsing Mpox SNPs...")
    assignments["mpox"] = parse_mpox()

    rows = []
    for virus, mapping in assignments.items():
        fasta = RAW / f"{virus}.fasta"
        for rec in SeqIO.parse(fasta, "fasta"):
            base = rec.id.split(".")[0]
            lin = mapping.get(base, "unknown")
            if lin != "unknown":
                rows.append({"accession": rec.id, "virus": virus, "lineage": lin, "length": len(rec.seq), "fasta": str(fasta)})
                
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "master_lineage_6v.csv", index=False)
    
    print("\n=== FINAL LINEAGE COUNTS (6 Viruses) ===")
    print(pd.crosstab(df["virus"], df["lineage"]).to_string())

if __name__ == "__main__":
    main()
