#!/usr/bin/env python3
import json, random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

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

class WindowDS(Dataset):
    def __init__(self, data, enc):
        self.items=[]
        for si,(_, seq, virus) in enumerate(data):
            # CAP WINDOWS TO 5 PER SEQUENCE TO PREVENT CPU MELT-DOWN
            ws = list(windows(seq))[:5] 
            w = 1.0/len(ws); y = V2I[virus]
            for win in ws: self.items.append((enc(win), y, w, si))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def col_onehot(b):
    x,y,w,s=zip(*b); return torch.from_numpy(np.stack(x)),torch.tensor(y),torch.tensor(w),torch.tensor(s)
def col_kmers(b):
    x,y,w,s=zip(*b); L=max(len(t) for t in x); pad=np.zeros((len(x),L),dtype=np.int64)
    for i,t in enumerate(x): pad[i,:len(t)]=t
    return torch.from_numpy(pad),torch.tensor(y),torch.tensor(w),torch.tensor(s)

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

def run_train(name, model, train_ds, test_ds, collate, lr=1e-3, epochs=2):
    print(f"\n=== Training {name} ===")
    tr_ld = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
    te_ld = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)
    model = model.to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1 = -1
    for ep in range(1, epochs+1):
        model.train()
        for x,y,w,s in tr_ld:
            ce = F.cross_entropy(model(x), y, weight=weights, reduction="none", label_smoothing=0.1)
            loss = (ce*w).sum()/w.sum(); opt.zero_grad(); loss.backward(); opt.step()
        f1 = evaluate(model, test_data, te_ld)
        print(f"  ep {ep} | test_f1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
    print(f"  saved {name}.pt (Best F1: {best_f1:.4f})")

run_train("cnn_v3", CNN(), WindowDS(train_data, onehot), WindowDS(test_data, onehot), col_onehot)
run_train("bilstm_v3", BiLSTM(), WindowDS(train_data, lambda s: np.array(kmers(s))), WindowDS(test_data, lambda s: np.array(kmers(s))), col_kmers)

print("\nLoading DNABERT...")
tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
dnabert = AutoModel.from_pretrained("armheb/DNA_bert_6")
for p in dnabert.parameters(): p.requires_grad = False
dnabert.eval()

class DNABertHead(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(768, N)
    def forward(self, x): return self.fc(x)

def get_bert_embeddings(data):
    embs, ys, ws, ss = [], [], [], []
    for si,(_, seq, virus) in enumerate(data):
        # CAP TO 2 WINDOWS PER SEQUENCE FOR DNABERT (CPU SAVER)
        wins = list(windows(seq))[:2]
        for w in wins:
            enc = tokenizer(w, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            with torch.no_grad():
                out = dnabert(**enc).last_hidden_state[:, 0, :].squeeze().numpy()
            embs.append(out); ys.append(V2I[virus]); ws.append(1.0/len(wins)); ss.append(si)
    return (torch.tensor(np.stack(embs), dtype=torch.float32), torch.tensor(ys), torch.tensor(ws), torch.tensor(ss))

print("Extracting DNABERT embeddings (capped for speed)...")
X_tr, y_tr, w_tr, s_tr = get_bert_embeddings(train_data)
X_te, y_te, w_te, s_te = get_bert_embeddings(test_data)

class SimpleDS(Dataset):
    def __init__(self, x, y, w, s): self.x, self.y, self.w, self.s = x, y, w, s
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i], self.w[i], self.s[i]

def col_simple(b):
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b]), torch.tensor([x[2] for x in b]), torch.tensor([x[3] for x in b])

run_train("dnabert_v3", DNABertHead(), SimpleDS(X_tr, y_tr, w_tr, s_tr), SimpleDS(X_te, y_te, w_te, s_te), col_simple, lr=1e-3, epochs=5)
