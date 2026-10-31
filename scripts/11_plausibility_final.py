#!/usr/bin/env python3
import random, subprocess, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from Bio import SeqIO

REFS = {
    "dengue":      "NC_001477.1",
    "chikungunya": "NC_004162.2",
    "norovirus":   "NC_029646.1",
    "measles":     "NC_001498.1",
    "mpox":        "NC_063383.1",
    "adeno_c":     "NC_001405.1"
}

# Broader search terms
TARGETS = {
    "dengue":      {"antigenic": ["E", "ENV", "ENVELOPE"], "conserved": ["NS5", "POL"]},
    "chikungunya": {"antigenic": ["E1"], "conserved": ["NSP1"]},
    "norovirus":   {"antigenic": ["VP1", "CAPSID"], "conserved": ["RDRP", "POL"]},
    "measles":     {"antigenic": ["H", "HA", "F"], "conserved": ["L", "N"]},
    "mpox":        {"antigenic": ["B6R", "A29", "VIRION"], "conserved": ["DNA_POL", "POL"]},
    "adeno_c":     {"antigenic": ["HEXON", "FIBER"], "conserved": ["POL", "POLYMERASE"]}
}

def fetch_ref_genes(virus):
    cache = Path(f"data/raw/genbank/{virus}_ref.gb")
    if not cache.exists() or cache.stat().st_size == 0:
        print(f"Downloading {virus} reference...")
        # Increase timeout to 180s for Mpox
        r = subprocess.run(["efetch", "-db", "nuccore", "-id", REFS[virus], "-format", "gb"], capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not r.stdout.strip(): return {}
        cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(r.stdout)
    
    genes = defaultdict(list)
    try:
        for rec in SeqIO.parse(str(cache), "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS", "mat_peptide", "gene"): continue
                name = None
                for key in ("gene", "product", "standard_name"):
                    if key in feat.qualifiers: name = feat.qualifiers[key][0].upper(); break
                if not name: continue
                
                # Store both the raw name and the span
                try: genes[name].append((int(feat.location.start), int(feat.location.end)))
                except Exception:
                    for part in feat.location.parts: genes[name].append((int(part.start), int(part.end)))
    except Exception: return {}
    return dict(genes)

def get_oe_from_reference(attr, genes, target_names, seq_len, ref_len):
    n_bins = len(attr)
    k = max(1, int(0.10 * n_bins))
    top_bins = np.argsort(-np.abs(attr))[:k]
    
    scale = seq_len / ref_len
    # Broad match: check if ANY of the target names are a substring of the gene name
    scaled_spans = []
    for g, sp in genes.items():
        for target in target_names:
            if target in g:
                scaled_spans.extend([(int(s*scale), int(e*scale)) for s, e in sp])
                break
                
    if not scaled_spans: return None
    
    hits = 0
    for b in top_bins:
        bs, be = b*100, min((b+1)*100, seq_len)
        if any(bs < e and be > s for s, e in scaled_spans): hits += 1
            
    target_len = sum(e-s for s,e in scaled_spans)
    exp = k * (target_len / seq_len)
    return (hits / exp) if exp > 0 else None

print("Loading test data...")
test_data = [(rec.id.split("|")[-1], str(rec.seq).upper(), rec.id.split("|")[0]) for rec in SeqIO.parse("data/processed/final_split/test.fasta", "fasta")]
by_virus = defaultdict(list)
for rec in test_data: by_virus[rec[2]].append(rec)

print("\n--- PLAUSIBILITY (O/E vs Reference Genome) ---")
print(f"{'Virus':12s} | {'Antigenic O/E':15s} | {'Conserved O/E':15s} | Genes Found")
print("-" * 75)

for v in ["dengue", "chikungunya", "norovirus", "measles", "mpox", "adeno_c"]:
    genes = fetch_ref_genes(v)
    if not genes:
        print(f"{v:12s} | No reference annotation found")
        continue
        
    rng = random.Random(42); rng.shuffle(by_virus[v])
    samples = by_virus[v][:3]
    
    # Get the reference length
    ref_len = 0
    for rec in SeqIO.parse(f"data/raw/genbank/{v}_ref.gb", "genbank"):
        ref_len = len(rec.seq); break
        
    ant_oes, con_oes = [], []
    for acc, seq, _ in samples:
        # Fake attribution (since we just want to test the gene mapping)
        fake_attr = np.random.rand(len(seq) // 100) 
        
        oe_a = get_oe_from_reference(fake_attr, genes, TARGETS[v]["antigenic"], len(seq), ref_len)
        oe_c = get_oe_from_reference(fake_attr, genes, TARGETS[v]["conserved"], len(seq), ref_len)
        if oe_a is not None: ant_oes.append(oe_a)
        if oe_c is not None: con_oes.append(oe_c)
        
    med_a = np.median(ant_oes) if ant_oes else None
    med_c = np.median(con_oes) if con_oes else None
    a_str = f"{med_a:.2f}" if med_a is not None else "N/A"
    c_str = f"{med_c:.2f}" if med_c is not None else "N/A"
    
    # Print the first 4 gene names found to help debug
    found_genes = list(genes.keys())[:4]
    print(f"{v:12s} | {a_str:15s} | {c_str:15s} | {', '.join(found_genes)}")

print("-" * 75)
