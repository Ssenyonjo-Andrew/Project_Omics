#!/usr/bin/env python3
import json, random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from scipy.stats import spearmanr

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
V2I = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
WINDOW = 3000; STRIDE = 1500; BIN_BP = 100; DEV = "cpu"

# --- Models ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 64, 8, padding="same"); self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 128, 8, padding="same"); self.b2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(0.4); self.fc = nn.Linear(128, N)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4); self.fc = nn.Linear(128, N)
    def forward(self, x):
        o,_ = self.lstm(self.emb(x)); return self.fc(self.drop(o.mean(1)))

class GenomicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 128, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(128, N)
    def forward(self, x):
        e = self.emb(x); out = self.transformer(e)
        return self.fc(out.mean(dim=1))

# --- Load Models ---
cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn_v3.pt", map_location=DEV)); cnn.eval()
bil = BiLSTM(); bil.load_state_dict(torch.load("models/bilstm_v3.pt", map_location=DEV)); bil.eval()
trf = GenomicTransformer(); trf.load_state_dict(torch.load("models/transformer_v3.pt", map_location=DEV)); trf.eval()

# --- Helpers ---
def windows(seq):
    if len(seq) < WINDOW: yield seq + "N"*(WINDOW-len(seq)); return
    for i in range(0, len(seq)-WINDOW+1, STRIDE): yield seq[i:i+WINDOW]
    if (len(seq)-WINDOW) % STRIDE: yield seq[-WINDOW:]

DNA = {"A":0,"C":1,"G":2,"T":3}
def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i,c in enumerate(s):
        if c in DNA: x[DNA[c],i]=1.0
    return x

def kmers(s, k=6):
    ids=[]
    for i in range(0,len(s)-k+1,k):
        idx,ok=0,True
        for c in s[i:i+k]:
            if c not in "ACGT": ok=False; break
            idx=idx*4+"ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]

@torch.no_grad()
def predict(seq, model_type):
    ws = list(windows(seq))
    if model_type == "cnn":
        X = torch.stack([torch.from_numpy(onehot(w)) for w in ws])
        p = F.softmax(cnn(X), -1).mean(0)
    elif model_type == "bilstm":
        toks = [kmers(w) for w in ws]; L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i,t in enumerate(toks): X[i,:len(t)] = torch.tensor(t)
        p = F.softmax(bil(X), -1).mean(0)
    else:
        toks = [kmers(w) for w in ws]; L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i,t in enumerate(toks): X[i,:len(t)] = torch.tensor(t)
        p = F.softmax(trf(X), -1).mean(0)
    return p.numpy()

@torch.no_grad()
def get_attribution(seq, model_type, cls):
    n_bins = len(seq) // BIN_BP
    variants = []
    for b in range(n_bins):
        s, e = b*BIN_BP, min((b+1)*BIN_BP, len(seq))
        variants.append(seq[:s] + "N"*(e-s) + seq[e:])
        
    base_p = predict(seq, model_type)[cls]
    probs = []
    for i in range(0, len(variants), 32): # Batch inference
        batch = variants[i:i+32]
        for v in batch:
            probs.append(predict(v, model_type)[cls])
    return base_p - np.array(probs)

def rank(arr):
    order = np.argsort(-np.abs(arr))
    r = np.empty(len(arr), dtype=float)
    r[order] = np.arange(1, len(arr)+1)
    return r

# --- Main ---
print("Loading test data...")
test_data = []
for rec in SeqIO.parse("data/processed/final_split/test.fasta", "fasta"):
    virus = rec.id.split("|")[0]
    test_data.append((rec.id, str(rec.seq).upper(), virus))

# Sample 5 sequences per virus
samples = []
by_virus = {v: [] for v in VIRUSES}
for rec in test_data: by_virus[rec[2]].append(rec)
for v in VIRUSES:
    rng = random.Random(SEED)
    rng.shuffle(by_virus[v])
    samples.extend(by_virus[v][:5])

print(f"Evaluating {len(samples)} sequences (5 per virus)...")

results = {"stability": [], "fidelity": []}

for idx, (acc, seq, true_virus) in enumerate(samples):
    print(f"[{idx+1}/{len(samples)}] {acc} ({true_virus})")
    
    # 1. Predictions & Attributions
    attrs = {}
    for m in ["cnn", "bilstm", "transformer"]:
        p = predict(seq, m)
        cls = int(np.argmax(p))
        attr = get_attribution(seq, m, cls)
        attrs[m] = attr

    # 2. Stability (Spearman correlations between models)
    rho_cb = spearmanr(attrs["cnn"], attrs["bilstm"]).correlation
    rho_ct = spearmanr(attrs["cnn"], attrs["transformer"]).correlation
    rho_bt = spearmanr(attrs["bilstm"], attrs["transformer"]).correlation
    
    if not (np.isnan(rho_cb) or np.isnan(rho_ct) or np.isnan(rho_bt)):
        results["stability"].append({"cnn_bilstm": rho_cb, "cnn_trf": rho_ct, "bilstm_trf": rho_bt})

    # 3. Fidelity (Consensus Top 10% deletion)
    consensus = (rank(attrs["cnn"]) + rank(attrs["bilstm"]) + rank(attrs["transformer"])) / 3.0
    n_drop = max(1, int(0.10 * len(consensus)))
    top_bins = np.argsort(consensus)[:n_drop]
    
    # Mask top bins
    seq_list = list(seq)
    for b in top_bins:
        s, e = b*BIN_BP, min((b+1)*BIN_BP, len(seq))
        for i in range(s, e): seq_list[i] = "N"
    masked_seq = "".join(seq_list)

    for m in ["cnn", "bilstm", "transformer"]:
        p_base = predict(seq, m)
        p_mask = predict(masked_seq, m)
        true_cls = V2I[true_virus]
        drop = p_base[true_cls] - p_mask[true_cls]
        results["fidelity"].append({"model": m, "virus": true_virus, "drop": float(drop)})

# --- Aggregate & Print ---
print("\n" + "="*50)
print("FINAL XAI EVALUATION RESULTS")
print("="*50)

s_df = pd.DataFrame(results["stability"])
f_df = pd.DataFrame(results["fidelity"])

print("\n--- 1. Cross-Architecture Stability (Median Spearman ρ) ---")
print(f"CNN vs BiLSTM:        {s_df['cnn_bilstm'].median():.3f}")
print(f"CNN vs Transformer:   {s_df['cnn_trf'].median():.3f}")
print(f"BiLSTM vs Transformer:{s_df['bilstm_trf'].median():.3f}")

print("\n--- 2. Fidelity (Margin Drop at 10% Deletion) ---")
print(f_df.groupby("model")["drop"].median().to_string())

with open("results/final_xai_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/final_xai_results.json")
