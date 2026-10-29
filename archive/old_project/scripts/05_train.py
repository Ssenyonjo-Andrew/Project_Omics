"""Phase 3: train CNN, BiLSTM, DNABERT-2 (linear probe) on the balanced benchmark."""
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio import SeqIO
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

CFG = yaml.safe_load(open("config.yaml"))
SEED = CFG["project"]["seed"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

FAMILIES = CFG["models"]["families"]
FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N_CLASSES = len(FAMILIES)
CKPT_DIR = Path(CFG["training"]["checkpoint_dir"]); CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR  = Path(CFG["training"]["results_dir"]);    RES_DIR.mkdir(parents=True, exist_ok=True)
DEV = ("cuda" if torch.cuda.is_available() else "cpu") if CFG["training"]["device"] == "auto" else CFG["training"]["device"]
print(f"device: {DEV}")


def load_split():
    df = pd.read_csv("data/processed/master_final.csv")
    cache = {}; out = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        p = row["sequence_file_path"]
        if p not in cache:
            cache[p] = {r.id.split(".")[0]: str(r.seq) for r in SeqIO.parse(p, "fasta")}
        seq = cache[p].get(row["accession"].split(".")[0])
        if seq:
            out[row["split"]].append((row["accession"], seq.upper(), row["family"]))
    for k, v in out.items():
        print(f"  {k}: {len(v)} sequences, {len({x[2] for x in v})} families")
    return out


def windows(seq, wlen, stride):
    if len(seq) < wlen:
        yield seq + "N" * (wlen - len(seq)); return
    for i in range(0, len(seq) - wlen + 1, stride):
        yield seq[i:i + wlen]
    if (len(seq) - wlen) % stride:
        yield seq[-wlen:]


DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
def onehot(seq):
    x = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in DNA_MAP: x[DNA_MAP[c], i] = 1.0
    return x


def kmers(seq, k):
    ids = []
    for i in range(0, len(seq) - k + 1, k):
        idx, ok = 0, True
        for c in seq[i:i + k]:
            if c not in "ACGT": ok = False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]


class WindowDataset(Dataset):
    def __init__(self, records, encoder, wlen, stride):
        self.items = []
        for sidx, (_, seq, fam) in enumerate(records):
            wins = list(windows(seq, wlen, stride))
            w = 1.0 / max(len(wins), 1); y = FAM2IDX[fam]
            for win in wins:
                self.items.append((encoder(win), y, w, sidx))
        print(f"    {len(records)} sequences -> {len(self.items)} windows")
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_onehot(batch):
    xs, ys, ws, ss = zip(*batch)
    return (torch.from_numpy(np.stack(xs)), torch.tensor(ys), torch.tensor(ws), torch.tensor(ss))


def collate_kmers(batch):
    xs, ys, ws, ss = zip(*batch)
    L = max(len(x) for x in xs)
    pad = np.zeros((len(xs), L), dtype=np.int64)
    for i, x in enumerate(xs): pad[i, :len(x)] = x
    return (torch.from_numpy(pad), torch.tensor(ys), torch.tensor(ws), torch.tensor(ss))


class CNN(nn.Module):
    def __init__(self, f1, f2, k, dr):
        super().__init__()
        self.c1 = nn.Conv1d(4, f1, k, padding="same"); self.b1 = nn.BatchNorm1d(f1)
        self.c2 = nn.Conv1d(f1, f2, k, padding="same"); self.b2 = nn.BatchNorm1d(f2)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(f2, N_CLASSES)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab, emb, hid, dr):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dr); self.fc = nn.Linear(hid * 2, N_CLASSES)
    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(self.drop(out.mean(dim=1)))


def train_epoch(model, loader, opt):
    model.train(); tot, n = 0.0, 0
    for x, y, w, s in loader:
        x, y, w = x.to(DEV), y.to(DEV), w.to(DEV)
        ce = F.cross_entropy(model(x), y, reduction="none")
        loss = (ce * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(torch.unique(s)); n += len(torch.unique(s))
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, records):
    model.eval(); probs = [[] for _ in records]
    for x, y, w, s in loader:
        p = F.softmax(model(x.to(DEV)), -1).cpu().numpy()
        for i, sidx in enumerate(s.tolist()): probs[sidx].append(p[i])
    yt, yp = [], []
    for sidx, (_, _, fam) in enumerate(records):
        if not probs[sidx]: continue
        yt.append(FAM2IDX[fam]); yp.append(int(np.argmax(np.mean(np.stack(probs[sidx]), 0))))
    macro = f1_score(yt, yp, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=range(N_CLASSES), zero_division=0)
    return dict(macro_f1=float(macro), per_class_f1=[float(v) for v in f1], n=len(yt))


def run_training(name, model, train_rec, val_rec, ds_fn, collate, lr, bs, epochs, patience):
    print(f"\n=== {name} ===")
    tr = DataLoader(ds_fn(train_rec), batch_size=bs, shuffle=True,  collate_fn=collate)
    va = DataLoader(ds_fn(val_rec),   batch_size=bs, shuffle=False, collate_fn=collate)
    model = model.to(DEV); opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, tr, opt); m = evaluate(model, va, val_rec)
        print(f"  ep {ep:2d}  loss {loss:.4f}  val_macro_f1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1, wait = m["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: print(f"  early stop at epoch {ep}"); break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CKPT_DIR / f"{name}.pt")
    print(f"  saved {CKPT_DIR}/{name}.pt  (best val macro-F1 = {best_f1:.4f})")
    return best_f1


def main():
    print("loading data...")
    splits = load_split(); results = {}

    c = CFG["models"]["cnn"]
    f1 = run_training("cnn", CNN(c["filters"][0], c["filters"][1], c["kernel"], c["dropout"]),
                      splits["train"], splits["val"],
                      lambda r: WindowDataset(r, onehot, c["window"], c["stride"]),
                      collate_onehot, c["lr"], c["batch_size"], c["epochs"], c["patience"])
    results["cnn"] = {"val_macro_f1": f1}

    b = CFG["models"]["bilstm"]
    f1 = run_training("bilstm", BiLSTM(b["vocab"], b["embed"], b["hidden"], b["dropout"]),
                      splits["train"], splits["val"],
                      lambda r: WindowDataset(r, lambda s: np.array(kmers(s, b["kmer"])), b["window"], b["stride"]),
                      collate_kmers, b["lr"], b["batch_size"], b["epochs"], b["patience"])
    results["bilstm"] = {"val_macro_f1": f1}

    from transformers import AutoTokenizer, AutoModel
    d = CFG["models"]["dnabert2"]
    print("\n=== dnabert2 (linear probe on frozen BERT) ===")
    tok  = AutoTokenizer.from_pretrained(d["name"], trust_remote_code=True)
    bert = AutoModel.from_pretrained(d["name"], trust_remote_code=True).to(DEV)
    for p_ in bert.parameters(): p_.requires_grad = False
    bert.eval()

    @torch.no_grad()
    def embed(records):
        embs, ys, ws, ss = [], [], [], []
        for sidx, (_, seq, fam) in enumerate(records):
            wins = list(windows(seq, d["chunk_bp"], d["stride_bp"]))
            w = 1.0 / max(len(wins), 1); y = FAM2IDX[fam]
            for win in wins:
                enc = tok(win, return_tensors="pt", max_length=d["max_tokens"],
                          truncation=True, padding="max_length").to(DEV)
                h = bert(**enc).last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(h[0]); ys.append(y); ws.append(w); ss.append(sidx)
        return np.stack(embs).astype(np.float32), np.array(ys), np.array(ws), np.array(ss)

    print("  extracting train embeddings...")
    Xtr, ytr, wtr, _ = embed(splits["train"]); print(f"    train windows: {len(Xtr)}")
    print("  extracting val embeddings...")
    Xva, yva, wva, sva = embed(splits["val"]);   print(f"    val windows: {len(Xva)}")

    head = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, N_CLASSES)).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best_f1, best_state = -1.0, None
    for ep in range(1, 31):
        head.train(); perm = np.random.permutation(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]
            xb = torch.from_numpy(Xtr[idx]).to(DEV); yb = torch.from_numpy(ytr[idx]).to(DEV); wb = torch.from_numpy(wtr[idx]).to(DEV)
            ce = F.cross_entropy(head(xb), yb, reduction="none")
            loss = (ce * wb).sum() / wb.sum(); opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(Xva).to(DEV)).cpu().numpy()
        by_seq = {}
        for i, sidx in enumerate(sva.tolist()): by_seq.setdefault(sidx, []).append(logits[i])
        yt, yp = [], []
        for sidx, (_, _, fam) in enumerate(splits["val"]):
            if sidx not in by_seq: continue
            yt.append(FAM2IDX[fam]); yp.append(int(np.argmax(np.mean(np.stack(by_seq[sidx]), 0))))
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        if ep % 5 == 0 or ep == 30: print(f"  ep {ep:2d}  val_macro_f1 {f1:.4f}")
        if f1 > best_f1: best_f1 = f1; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    torch.save({"head": head.state_dict(), "bert_name": d["name"]}, CKPT_DIR / "dnabert2_probe.pt")
    results["dnabert2"] = {"val_macro_f1": float(best_f1), "mode": "linear_probe_frozen_bert"}
    print(f"  saved {CKPT_DIR}/dnabert2_probe.pt  (best val macro-F1 = {best_f1:.4f})")

    maj = 0.20
    for name, r in results.items(): r["passes_gate"] = r.get("val_macro_f1", 0) > maj
    with open(RES_DIR / "train_metrics.json", "w") as f: json.dump(results, f, indent=2)
    print(f"\nWrote {RES_DIR}/train_metrics.json")
    for name, r in results.items():
        print(f"  {name:10s}  val_macro_f1 = {r['val_macro_f1']:.4f}  gate = {r['passes_gate']}")


if __name__ == "__main__":
    main()
