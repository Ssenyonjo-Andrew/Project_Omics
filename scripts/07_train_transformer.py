#!/usr/bin/env python3
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
V2I = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
WINDOW = 3000; STRIDE = 1500; DEV = "cpu"
CKPT = Path("models"); CKPT.mkdir(exist_ok=True)

def load_data(path):
    data = []
    for rec in SeqIO.parse(path, "fasta"):
        virus = rec.id.split("|")[0]
        data.append((rec.id, str(rec.seq).upper(), virus))
    return data

train_data = load_data("data/processed/final_split/train.fasta")
test_data  = load_data("data/processed/final_split/test.fasta")
counts = Counter(x[2] for x in train_data)
weights = torch.tensor([len(train_data) / (N * counts.get(v, 1)) for v in VIRUSES], dtype=torch.float32)

def windows(seq):
    if len(seq) < WINDOW: yield seq + "N"*(WINDOW-len(seq)); return
    for i in range(0, len(seq)-WINDOW+1, STRIDE): yield seq[i:i+WINDOW]
    if (len(seq)-WINDOW) % STRIDE: yield seq[-WINDOW:]

def kmers(s, k=6):
    ids=[]
    for i in range(0,len(s)-k+1,k):
        idx,ok=0,True
        for c in s[i:i+k]:
            if c not in "ACGT": ok=False; break
            idx=idx*4+"ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]

class WindowDS(Dataset):
    def __init__(self, data):
        self.items=[]
        for si,(_, seq, virus) in enumerate(data):
            ws = list(windows(seq))[:5] # Cap at 5 for speed
            w = 1.0/len(ws); y = V2I[virus]
            for win in ws: self.items.append((kmers(win), y, w, si))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def col_kmers(b):
    x,y,w,s=zip(*b); L=max(len(t) for t in x); pad=np.zeros((len(x),L),dtype=np.int64)
    for i,t in enumerate(x): pad[i,:len(t)]=t
    return torch.from_numpy(pad),torch.tensor(y),torch.tensor(w),torch.tensor(s)

class GenomicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 128, padding_idx=0)
        # PyTorch Native Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, N)
        
    def forward(self, x):
        e = self.emb(x)
        out = self.transformer(e)
        return self.fc(out.mean(dim=1))

def evaluate(model, data, loader):
    model.eval(); probs=[[] for _ in data]
    with torch.no_grad():
        for x,y,w,s in loader:
            p = F.softmax(model(x), -1).numpy()
            for i,si in enumerate(s.tolist()): probs[si].append(p[i])
    yt,yp=[],[]
    for si,(_,_,v) in enumerate(data):
        if not probs[si]: continue
        yt.append(V2I[v]); yp.append(int(np.argmax(np.mean(np.stack(probs[si]),0))))
    return f1_score(yt, yp, average="macro", zero_division=0)

print("=== Training Transformer (Attention-based) ===")
tr_ld = DataLoader(WindowDS(train_data), batch_size=128, shuffle=True, collate_fn=col_kmers)
te_ld = DataLoader(WindowDS(test_data), batch_size=128, shuffle=False, collate_fn=col_kmers)

model = GenomicTransformer().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
best_f1 = -1

for ep in range(1, 4): # 3 epochs is enough
    model.train()
    for x,y,w,s in tr_ld:
        ce = F.cross_entropy(model(x), y, weight=weights, reduction="none", label_smoothing=0.1)
        loss = (ce*w).sum()/w.sum(); opt.zero_grad(); loss.backward(); opt.step()
    f1 = evaluate(model, test_data, te_ld)
    print(f"  ep {ep} | test_f1 {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), CKPT / "transformer_v3.pt")

print(f"  saved transformer_v3.pt (Best F1: {best_f1:.4f})")
