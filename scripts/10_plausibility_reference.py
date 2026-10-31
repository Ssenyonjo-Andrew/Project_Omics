#!/usr/bin/env python3
import random, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from Bio import SeqIO

# Reference GenBank accessions (ICTV exemplars)
REFS = {
    "dengue":      "NC_001477.1", # Dengue virus 1
    "chikungunya": "NC_004162.2", # Chikungunya
    "norovirus":   "NC_029646.1", # Norovirus GI
    "measles":     "NC_001498.1", # Measles
    "mpox":        "NC_063383.1", # Mpox
    "adeno_c":     "NC_001405.1"  # Human adenovirus C
}

TARGETS = {
    "dengue":      {"antigenic": ["E", "ENV"], "conserved": ["NS5"]},
    "chikungunya": {"antigenic": ["E1"], "conserved": ["NSP1"]},
    "norovirus":   {"antigenic": ["VP1", "CAPSID"], "conserved": ["RDRP"]},
    "measles":     {"antigenic": ["H", "HA", "F"], "conserved": ["L", "N"]},
    "mpox":        {"antigenic": ["B6R", "A29"], "conserved": ["DNA_POL", "POL"]},
    "adeno_c":     {"antigenic": ["HEXON", "FIBER"], "conserved": ["POL", "POLYMERASE"]}
}

def fetch_ref_genes(virus):
    cache = Path(f"data/raw/genbank/{virus}_ref.gb")
    if not cache.exists() or cache.stat().st_size == 0:
        r = subprocess.run(["efetch", "-db", "nuccore", "-id", REFS[virus], "-format", "gb"], capture_output=True, text=True, timeout=60)
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
                try: genes[name].append((int(feat.location.start), int(feat.location.end)))
                except Exception:
                    for part in feat.location.parts: genes[name].append((int(part.start), int(part.end)))
    except Exception: return {}
    return dict(genes)

def get_oe_from_reference(attr, genes, target_names, seq_len, ref_len):
    n_bins = len(attr)
    k = max(1, int(0.10 * n_bins))
    top_bins = np.argsort(-np.abs(attr))[:k]
    
    # Scale reference coordinates to test sequence length
    scale = seq_len / ref_len
    scaled_spans = [(int(s*scale), int(e*scale)) for g, sp in genes.items() for name in target_names if name in g for s, e in sp]
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
print(f"{'Virus':12s} | {'Antigenic O/E':15s} | {'Conserved O/E':15s}")
print("-" * 55)

for v in ["dengue", "chikungunya", "norovirus", "measles", "mpox", "adeno_c"]:
    genes = fetch_ref_genes(v)
    if not genes:
        print(f"{v:12s} | No reference annotation found")
        continue
        
    rng = random.Random(42); rng.shuffle(by_virus[v])
    samples = by_virus[v][:3] # 3 sequences per virus
    
    ant_oes, con_oes = [], []
    for acc, seq, _ in samples:
        # Simulate model attribution (random noise for this test, since we just want to check gene mapping)
        # In reality, this would be the occlusion scores from your CNN
        fake_attr = np.random.rand(len(seq) // 100) 
        
        ref_len = 0
        for rec in SeqIO.parse(f"data/raw/genbank/{v}_ref.gb", "genbank"):
            ref_len = len(rec.seq); break
            
        oe_a = get_oe_from_reference(fake_attr, genes, TARGETS[v]["antigenic"], len(seq), ref_len)
        oe_c = get_oe_from_reference(fake_attr, genes, TARGETS[v]["conserved"], len(seq), ref_len)
        if oe_a: ant_oes.append(oe_a)
        if oe_c: con_oes.append(oe_c)
        
    med_a = np.median(ant_oes) if ant_oes else None
    med_c = np.median(con_oes) if con_oes else None
    a_str = f"{med_a:.2f}" if med_a is not None else "N/A"
    c_str = f"{med_c:.2f}" if med_c is not None else "N/A"
    print(f"{v:12s} | {a_str:15s} | {c_str:15s}")

print("-" * 55)
print("Note: This uses ICTV/RefSeq exemplar genomes to map gene boundaries.")
