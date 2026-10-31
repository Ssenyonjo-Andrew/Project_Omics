#!/usr/bin/env python3
import random, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from scipy.stats import spearmanr

# --- Models ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 64, 8, padding="same"); self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 128, 8, padding="same"); self.b2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(0.4); self.fc = nn.Linear(128, 6)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4); self.fc = nn.Linear(128, 6)
    def forward(self, x):
        o,_ = self.lstm(self.emb(x)); return self.fc(self.drop(o.mean(1)))

class GenomicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 128, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(128, 6)
    def forward(self, x):
        e = self.emb(x); out = self.transformer(e)
        return self.fc(out.mean(dim=1))

DEV = "cpu"
cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn_v3.pt", map_location=DEV)); cnn.eval()
bil = BiLSTM(); bil.load_state_dict(torch.load("models/bilstm_v3.pt", map_location=DEV)); bil.eval()
trf = GenomicTransformer(); trf.load_state_dict(torch.load("models/transformer_v3.pt", map_location=DEV)); trf.eval()

VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
V2I = {v: i for i, v in enumerate(VIRUSES)}

TARGETS = {
    "dengue":      {"antigenic": ["E", "ENV"], "conserved": ["NS5"]},
    "chikungunya": {"antigenic": ["E1"], "conserved": ["NSP1"]},
    "norovirus":   {"antigenic": ["VP1", "CAPSID"], "conserved": ["RDRP"]},
    "measles":     {"antigenic": ["H", "HA", "F"], "conserved": ["L", "N"]},
    "mpox":        {"antigenic": ["B6R", "A29"], "conserved": ["DNA_POL", "POL"]},
    "adeno_c":     {"antigenic": ["HEXON", "FIBER"], "conserved": ["POL", "POLYMERASE"]}
}

def windows(seq):
    if len(seq) < 3000: return [seq.ljust(3000, 'N')]
    return [seq[i:i+3000] for i in range(0, len(seq)-3000+1, 1500)]

def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i,c in enumerate(s):
        if c in "ACGT": x["ACGT".index(c), i] = 1.0
    return x

def kmers(s):
    ids = []
    for i in range(0, len(s)-6+1, 6):
        idx, ok = 0, True
        for c in s[i:i+6]:
            if c not in "ACGT": ok=False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]

@torch.no_grad()
def get_predictions(variants, model_type, n_wins_per_var):
    flat_ws = [w for v in variants for w in windows(v)]
    if model_type == "cnn":
        X = torch.stack([torch.from_numpy(onehot(w)) for w in flat_ws])
        probs = F.softmax(cnn(X), -1).numpy()
    else:
        toks = [kmers(w) for w in flat_ws]; L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i,t in enumerate(toks): X[i,:len(t)] = torch.tensor(t)
        model = bil if model_type == "bilstm" else trf
        probs = F.softmax(model(X), -1).numpy()
    
    out = []
    for i in range(len(variants)):
        out.append(probs[i*n_wins_per_var:(i+1)*n_wins_per_var].mean(0))
    return np.array(out)

def fetch_genes(acc):
    cache = Path(f"data/raw/genbank/{acc.split('.')[0]}.gb")
    if not cache.exists() or cache.stat().st_size == 0: return {}
    genes = defaultdict(list)
    try:
        for rec in SeqIO.parse(str(cache), "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS", "mat_peptide", "gene"): continue
                name = next((feat.qualifiers[k][0].upper() for k in ("gene", "product") if k in feat.qualifiers), None)
                if not name: continue
                try: genes[name].append((int(feat.location.start), int(feat.location.end)))
                except Exception: pass
    except Exception: return {}
    return dict(genes)

print("Running Fast Evaluation (Stability, Fidelity, Plausibility)...\n")
test_data = [(rec.id.split("|")[-1], str(rec.seq).upper(), rec.id.split("|")[0]) for rec in SeqIO.parse("data/processed/final_split/test.fasta", "fasta")]
by_v = defaultdict(list)
for rec in test_data: by_v[rec[2]].append(rec)

results_stab, results_fid, results_plau = [], [], []

for v in VIRUSES:
    acc, seq, _ = by_v[v][0]
    print(f"Evaluating {v} ({len(seq)} bp)...")
    n_bins = len(seq) // 100
    sample_bins = np.random.choice(n_bins, min(10, n_bins), replace=False)
    variants = [seq[:b*100] + "N"*100 + seq[(b+1)*100:] for b in sample_bins]
    
    attrs = {}
    for m in ["cnn", "bilstm", "transformer"]:
        base_p = get_predictions([seq], m, 1)[0][V2I[v]]
        var_p = get_predictions(variants, m, 1)[:, V2I[v]]
        attrs[m] = base_p - var_p

    # 1. Stability
    rho_cb = spearmanr(attrs["cnn"], attrs["bilstm"]).correlation
    rho_ct = spearmanr(attrs["cnn"], attrs["transformer"]).correlation
    rho_bt = spearmanr(attrs["bilstm"], attrs["transformer"]).correlation
    results_stab.append({"cnn_bilstm": rho_cb, "cnn_trf": rho_ct, "bilstm_trf": rho_bt})

    # 2. Fidelity (Delete top 3 sampled bins)
    consensus = (attrs["cnn"] + attrs["bilstm"] + attrs["transformer"]) / 3.0
    top_idx = np.argsort(-consensus)[:3]
    mask_seq = list(seq)
    for i in top_idx:
        b = sample_bins[i]
        for j in range(b*100, min((b+1)*100, len(seq))): mask_seq[j] = "N"
    mask_seq = "".join(mask_seq)
    
    drop_c = get_predictions([seq], "cnn", 1)[0][V2I[v]] - get_predictions([mask_seq], "cnn", 1)[0][V2I[v]]
    drop_b = get_predictions([seq], "bilstm", 1)[0][V2I[v]] - get_predictions([mask_seq], "bilstm", 1)[0][V2I[v]]
    drop_t = get_predictions([seq], "transformer", 1)[0][V2I[v]] - get_predictions([mask_seq], "transformer", 1)[0][V2I[v]]
    results_fid.append({"cnn": drop_c, "bilstm": drop_b, "transformer": drop_t})

    # 3. Plausibility
    genes = fetch_genes(acc)
    if genes:
        top_spans = [ (sample_bins[i]*100, (sample_bins[i]+1)*100) for i in top_idx ]
        ant_spans = [s for g, sp in genes.items() for name in TARGETS[v]["antigenic"] if name in g for s in sp]
        con_spans = [s for g, sp in genes.items() for name in TARGETS[v]["conserved"] if name in g for s in sp]
        
        def count_hits(spans):
            return sum(1 for bs, be in top_spans if any(bs < e and be > s for s, e in spans))
            
        h_ant = count_hits(ant_spans)
        h_con = count_hits(con_spans)
        
        exp_ant = 3 * (sum(e-s for s,e in ant_spans) / len(seq)) if ant_spans else 0
        exp_con = 3 * (sum(e-s for s,e in con_spans) / len(seq)) if con_spans else 0
        
        oe_a = h_ant / exp_ant if exp_ant > 0 else None
        oe_c = h_con / exp_con if exp_con > 0 else None
        results_plau.append({"virus": v, "oe_ant": oe_a, "oe_con": oe_c})

print("\n" + "="*65)
print("FINAL XAI EVALUATION RESULTS")
print("="*65)

s_df = pd.DataFrame(results_stab)
f_df = pd.DataFrame(results_fid)
p_df = pd.DataFrame(results_plau)

print("\n--- 1. Cross-Architecture Stability (Median Spearman ρ) ---")
print(f"CNN vs BiLSTM:         {s_df['cnn_bilstm'].median():.3f}")
print(f"CNN vs Transformer:    {s_df['cnn_trf'].median():.3f}")
print(f"BiLSTM vs Transformer: {s_df['bilstm_trf'].median():.3f}")

print("\n--- 2. Fidelity (Probability Drop when Top Bins Deleted) ---")
print(f"CNN:         {f_df['cnn'].median():+.4f}")
print(f"BiLSTM:      {f_df['bilstm'].median():+.4f}")
print(f"Transformer: {f_df['transformer'].median():+.4f}")

print("\n--- 3. Plausibility (O/E of Top Bins in Genes) ---")
if not p_df.empty:
    for _, row in p_df.iterrows():
        a = f"{row['oe_ant']:.2f}" if row['oe_ant'] is not None else "N/A"
        c = f"{row['oe_con']:.2f}" if row['oe_con'] is not None else "N/A"
        print(f"{row['virus']:12s} | Antigenic O/E: {a:5s} | Conserved O/E: {c:5s}")
else:
    print("No gene annotations available.")

print("\n(Note: Stability ρ < 0.3 means the models disagree. Fidelity > 0 means the explanation was faithful. Plausibility O/E > 1.0 means the model looked at the right gene.)")
