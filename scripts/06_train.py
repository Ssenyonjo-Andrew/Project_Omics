#!/usr/bin/env python3
"""Phase 6: train CNN + BiLSTM on 4-way virus classification."""
import json, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio import SeqIO
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

CFG = yaml.safe_load(open("config.yaml"))
SEED = CFG["project"]["seed"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

VIRUSES = ["dengue", "chikungunya", "hcv", "norovirus"]
VIR_TO_IDX = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
CKPT = Path("models"); CKPT.mkdir(exist_ok=True)
RES = Path("results"); RES.mkdir(exist_ok=True)
DEV = "cpu"


def load_labels():
    lab = {}
    for v in VIRUSES:
        df = pd.read_csv(f"data/processed/lineage/{v}_lineages.csv")
        for _, row in df.iterrows():
            acc = row["accession"].split(".")[0]
            lab[acc] = {"virus": v, "lineage": row["lineage"]}
    return lab


def load_fasta(path):
    out = []
    for rec in SeqIO.parse(path, "fasta"):
        base = rec.id.split(".")[0]
        lab = LABELS.get(base)
        if lab is None:
            continue
        out.append((rec.id, str(rec.seq).upper(),
                    lab["virus"], lab["lineage"]))
    return out


print("loading labels...")
LABELS = load_labels()
train = load_fasta("data/processed/split/all_train.fasta")
val   = load_fasta("data/processed/split/all_val.fasta")
test  = load_fasta("data/processed/split/all_test.fasta")
print(f"train={len(train)} val={len(val)} test={len(test)}")
print(f"train viruses: {dict(Counter(s[2] for s in train))}")
print(f"test viruses:  {dict(Counter(s[2] for s in test))}")

# class weights
counts = Counter(s[2] for s in train)
weights = torch.tensor(
    [len(train) / (N * counts.get(v, 1)) for v in VIRUSES],
    dtype=torch.float32
)
print(f"class weights: {dict(zip(VIRUSES, weights.tolist()))}")


def windows(seq, wlen, stride):
    if len(seq) < wlen:
        yield seq + "N" * (wlen - len(seq)); return
    for i in range(0, len(seq) - wlen + 1, stride):
        yield seq[i:i + wlen]
    if (len(seq) - wlen) % stride:
        yield seq[-wlen:]


DNA = {"A": 0, "C": 1, "G": 2, "T": 3}
def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in DNA: x[DNA[c], i] = 1.0
    return x


def kmers(s, k=6):
    ids = []
    for i in range(0, len(s) - k + 1, k):
        idx, ok = 0, True
        for c in s[i:i + k]:
            if c not in "ACGT": ok = False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]


class DS(Dataset):
    def __init__(self, recs, enc, wlen, stride, use_kmers=False):
        self.items = []
        for sidx, (_, seq, virus, _) in enumerate(recs):
            ws = list(windows(seq, wlen, stride))
            w = 1.0 / max(len(ws), 1)
            y = VIR_TO_IDX[virus]
            for win in ws:
                self.items.append((enc(win), y, w, sidx))
        self.use_kmers = use_kmers
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def col_onehot(batch):
    xs, ys, ws, ss = zip(*batch)
    return (torch.from_numpy(np.stack(xs)), torch.tensor(ys),
            torch.tensor(ws), torch.tensor(ss))


def col_kmers(batch):
    xs, ys, ws, ss = zip(*batch)
    L = max(len(x) for x in xs)
    pad = np.zeros((len(xs), L), dtype=np.int64)
    for i, x in enumerate(xs): pad[i, :len(x)] = x
    return (torch.from_numpy(pad), torch.tensor(ys),
            torch.tensor(ws), torch.tensor(ss))


class CNN(nn.Module):
    def __init__(self, f1=128, f2=256, k=8, dr=0.4):
        super().__init__()
        self.c1 = nn.Conv1d(4, f1, k, padding="same"); self.b1 = nn.BatchNorm1d(f1)
        self.c2 = nn.Conv1d(f1, f2, k, padding="same"); self.b2 = nn.BatchNorm1d(f2)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(f2, N)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab=4096, emb=128, hid=128, dr=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dr); self.fc = nn.Linear(hid * 2, N)
    def forward(self, x):
        o, _ = self.lstm(self.emb(x))
        return self.fc(self.drop(o.mean(1)))


@torch.no_grad()
def evaluate(model, loader, recs):
    model.eval()
    probs = [[] for _ in recs]
    for x, y, w, s in loader:
        p = F.softmax(model(x), -1).numpy()
        for i, si in enumerate(s.tolist()):
            probs[si].append(p[i])
    yt, yp = [], []
    for si, (_, _, virus, _) in enumerate(recs):
        if not probs[si]: continue
        yt.append(VIR_TO_IDX[virus])
        yp.append(int(np.argmax(np.mean(np.stack(probs[si]), 0))))
    macro = f1_score(yt, yp, average="macro", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=range(N)).tolist()
    return {"macro_f1": float(macro), "cm": cm, "n": len(yt)}


def train_one(name, model, train_loader, val_loader, val_recs,
              lr, epochs, patience, wd=1e-4, ls=0.1):
    print(f"\n=== {name} ===")
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y, w, s in train_loader:
            x, y, w = x.to(DEV), y.to(DEV), w.to(DEV)
            ce = F.cross_entropy(model(x), y, weight=weights.to(DEV),
                                 reduction="none", label_smoothing=ls)
            loss = (ce * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(torch.unique(s)); n += len(torch.unique(s))
        m = evaluate(model, val_loader, val_recs)
        print(f"  ep {ep:2d}  loss {tot/n:.4f}  val_macro_f1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1 + 1e-4:
            best_f1, wait = m["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {ep}"); break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CKPT / f"{name}.pt")
    print(f"  saved {CKPT}/{name}.pt  (val macro-F1 = {best_f1:.4f})")
    return model, best_f1


def main():
    # CNN
    tr_ds = DS(train, onehot, 3000, 1500)
    va_ds = DS(val, onehot, 3000, 1500)
    te_ds = DS(test, onehot, 3000, 1500)
    print(f"\nCNN windows: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}")

    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=col_onehot)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=col_onehot)
    te_ld = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=col_onehot)

    cnn, cnn_val = train_one("cnn_virus", CNN(), tr_ld, va_ld, val,
                            lr=1e-3, epochs=20, patience=5)
    cnn_test = evaluate(cnn, te_ld, test)
    print(f"  CNN test macro-F1 = {cnn_test['macro_f1']:.4f}  (n={cnn_test['n']})")
    print(f"  CNN test confusion (rows=true, cols=pred, order={VIRUSES}):")
    for row in cnn_test["cm"]: print(f"    {row}")

    # BiLSTM
    tr_ds = DS(train, lambda s: np.array(kmers(s)), 3000, 1500, use_kmers=True)
    va_ds = DS(val, lambda s: np.array(kmers(s)), 3000, 1500, use_kmers=True)
    te_ds = DS(test, lambda s: np.array(kmers(s)), 3000, 1500, use_kmers=True)
    print(f"\nBiLSTM windows: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}")

    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=col_kmers)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=col_kmers)
    te_ld = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=col_kmers)

    bil, bil_val = train_one("bilstm_virus", BiLSTM(), tr_ld, va_ld, val,
                            lr=1e-3, epochs=20, patience=5)
    bil_test = evaluate(bil, te_ld, test)
    print(f"  BiLSTM test macro-F1 = {bil_test['macro_f1']:.4f}  (n={bil_test['n']})")
    print(f"  BiLSTM test confusion (rows=true, cols=pred, order={VIRUSES}):")
    for row in bil_test["cm"]: print(f"    {row}")

    results = {
        "viruses": VIRUSES,
        "train_counts": dict(Counter(s[2] for s in train)),
        "val_counts":   dict(Counter(s[2] for s in val)),
        "test_counts":  dict(Counter(s[2] for s in test)),
        "cnn":   {"val_macro_f1": cnn_val, "test": cnn_test},
        "bilstm": {"val_macro_f1": bil_val, "test": bil_test},
        "class_weights": {v: float(w) for v, w in zip(VIRUSES, weights)},
    }
    json.dump(results, open(RES / "train_results.json", "w"), indent=2)
    print(f"\nwrote {RES}/train_results.json")


if __name__ == "__main__":
    main()
