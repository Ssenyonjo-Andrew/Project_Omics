#!/usr/bin/env python3
import random, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 64, 8, padding="same"); self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 128, 8, padding="same"); self.b2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(0.4); self.fc = nn.Linear(128, 6)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))

DEV = "cpu"
cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn_v3.pt", map_location=DEV)); cnn.eval()

VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
V2I = {v: i for i, v in enumerate(VIRUSES)}
REFS = {"dengue": "NC_001477.1", "chikungunya": "NC_004162.2", "norovirus": "NC_029646.1", "measles": "NC_001498.1", "mpox": "NC_063383.1", "adeno_c": "NC_001405.1"}

# PERFECTED TARGETS based on your GenBank output
TARGETS = {
    "dengue":      {"antigenic": ["E", "ENV", "POLY"], "conserved": ["NS5", "POLY"]},
    "chikungunya": {"antigenic": ["STRUCTURAL"], "conserved": ["NONSTRUCTURAL"]},
    "norovirus":   {"antigenic": ["ORF2", "VP1"], "conserved": ["ORF1", "RDRP"]},
    "measles":     {"antigenic": ["H", "F", "G"], "conserved": ["L", "N", "P"]},
    "mpox":        {"antigenic": ["OPG"], "conserved": ["OPG"]}, # Pox uses OPG
    "adeno_c":     {"antigenic": ["II", "III", "IV", "HEXON"], "conserved": ["E1A", "E1B", "POL"]}
}

def windows(seq):
    return [seq[i:i+3000] for i in range(0, len(seq)-3000+1, 1500)] or [seq.ljust(3000, 'N')]

def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i,c in enumerate(s):
        if c in "ACGT": x["ACGT".index(c), i] = 1.0
    return x

@torch.no_grad()
def get_attribution(seq, cls):
    n_bins = len(seq) // 100
    sample_bins = np.random.choice(n_bins, min(10, n_bins), replace=False)
    
    variants = [seq[:b*100] + "N"*100 + seq[(b+1)*100:] for b in sample_bins]
    flat_ws = [w for v in variants for w in windows(v)]
    X = torch.stack([torch.from_numpy(onehot(w)) for w in flat_ws])
    probs = F.softmax(cnn(X), -1).numpy()
    
    n_win = len(windows(seq))
    attr = np.zeros(10)
    for i in range(10):
        attr[i] = probs[i*n_win:(i+1)*n_win, cls].mean()
    return sample_bins, attr

def fetch_ref_genes(virus):
    cache = Path(f"data/raw/genbank/{virus}_ref.gb")
    if not cache.exists(): return {}
    genes = defaultdict(list)
    try:
        for rec in SeqIO.parse(str(cache), "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS", "mat_peptide", "gene"): continue
                name = next((feat.qualifiers[k][0].upper() for k in ("gene", "product") if k in feat.qualifiers), "")
                if not name: continue
                try: genes[name].append((int(feat.location.start), int(feat.location.end)))
                except Exception: pass
    except Exception: return {}
    return dict(genes)

print("Loading test data...")
test_data = [(rec.id.split("|")[-1], str(rec.seq).upper(), rec.id.split("|")[0]) for rec in SeqIO.parse("data/processed/final_split/test.fasta", "fasta")]
by_virus = defaultdict(list)
for rec in test_data: by_virus[rec[2]].append(rec)

print("\n--- FINAL PLAUSIBILITY (CNN Attribution vs Reference Genome) ---")
print(f"{'Virus':12s} | {'Antigenic O/E':15s} | {'Conserved O/E':15s}")
print("-" * 55)

for v in VIRUSES:
    genes = fetch_ref_genes(v)
    if not genes:
        print(f"{v:12s} | No reference annotation found")
        continue
        
    rng = random.Random(42); rng.shuffle(by_virus[v])
    acc, seq, _ = by_virus[v][0] # 1 sequence per virus for speed
    
    ref_len = 0
    for rec in SeqIO.parse(f"data/raw/genbank/{v}_ref.gb", "genbank"):
        ref_len = len(rec.seq); break
        
    sample_bins, attr = get_attribution(seq, V2I[v])
    
    def calc_oe(targets):
        scale = len(seq) / ref_len
        spans = [ (int(s*scale), int(e*scale)) for g, sp in genes.items() for t in targets if t in g for s, e in sp ]
        if not spans: return None
        hits = sum(1 for i, b in enumerate(sample_bins) if attr[i] > np.median(attr) and any(b*100 < e and (b+1)*100 > s for s, e in spans))
        exp = (len(sample_bins)/2) * (sum(e-s for s,e in spans) / len(seq))
        return (hits / exp) if exp > 0 else None

    oe_a = calc_oe(TARGETS[v]["antigenic"])
    oe_c = calc_oe(TARGETS[v]["conserved"])
    
    a_str = f"{oe_a:.2f}" if oe_a is not None else "N/A"
    c_str = f"{oe_c:.2f}" if oe_c is not None else "N/A"
    print(f"{v:12s} | {a_str:15s} | {c_str:15s}")

print("-" * 55)
print("Note: O/E > 1.0 means the CNN's top-attributed regions fall on these genes more than chance.")
