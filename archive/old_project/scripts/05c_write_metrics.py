"""Rebuild train_metrics.json from saved checkpoints."""
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio import SeqIO
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

CFG = yaml.safe_load(open("config.yaml"))
SEED = CFG["project"]["seed"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

FAMILIES = CFG["models"]["families"]
FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N_CLASSES = len(FAMILIES)
CKPT_DIR = Path(CFG["training"]["checkpoint_dir"])
RES_DIR  = Path(CFG["training"]["results_dir"]); RES_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cpu"


def load_split():
    df = pd.read_csv("data/processed/master_final.csv")
    cache = {}; out = {"train": [], "val": []}
    for _, r in df.iterrows():
        p = r["sequence_file_path"]
        if p not in cache:
            cache[p] = {rec.id.split(".")[0]: str(rec.seq) for rec in SeqIO.parse(p, "fasta")}
        seq = cache[p].get(r["accession"].split(".")[0])
        if seq and r["split"] in out:
            out[r["split"]].append((r["accession"], seq.upper(), r["family"]))
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
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_onehot(batch):
    xs, ys, ws, ss = zip(*batch)
    return (torch.from_numpy(np.stack(xs)), torch.tensor(ys),
            torch.tensor(ws), torch.tensor(ss))


def collate_kmers(batch):
    xs, ys, ws, ss = zip(*batch)
    L = max(len(x) for x in xs)
    pad = np.zeros((len(xs), L), dtype=np.int64)
    for i, x in enumerate(xs): pad[i, :len(x)] = x
    return (torch.from_numpy(pad), torch.tensor(ys),
            torch.tensor(ws), torch.tensor(ss))


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


@torch.no_grad()
def evaluate(model, loader, records):
    model.eval()
    probs = [[] for _ in records]
    for x, y, w, s in loader:
        p = F.softmax(model(x.to(DEV)), -1).cpu().numpy()
        for i, sidx in enumerate(s.tolist()): probs[sidx].append(p[i])
    yt, yp = [], []
    for sidx, (_, _, fam) in enumerate(records):
        if not probs[sidx]: continue
        yt.append(FAM2IDX[fam]); yp.append(int(np.argmax(np.mean(np.stack(probs[sidx]), 0))))
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def main():
    splits = load_split()
    c = CFG["models"]["cnn"]
    b = CFG["models"]["bilstm"]

    cnn = CNN(c["filters"][0], c["filters"][1], c["kernel"], c["dropout"]).to(DEV)
    cnn.load_state_dict(torch.load(CKPT_DIR / "cnn.pt", map_location=DEV))
    ds = WindowDataset(splits["val"], onehot, c["window"], c["stride"])
    cnn_f1 = evaluate(cnn, DataLoader(ds, batch_size=32, collate_fn=collate_onehot), splits["val"])

    bil = BiLSTM(b["vocab"], b["embed"], b["hidden"], b["dropout"]).to(DEV)
    bil.load_state_dict(torch.load(CKPT_DIR / "bilstm.pt", map_location=DEV))
    ds = WindowDataset(splits["val"], lambda s: np.array(kmers(s, b["kmer"])),
                       b["window"], b["stride"])
    bil_f1 = evaluate(bil, DataLoader(ds, batch_size=32, collate_fn=collate_kmers), splits["val"])

    metrics = {
        "cnn":    {"val_macro_f1": cnn_f1, "passes_gate": cnn_f1 > 0.20},
        "bilstm": {"val_macro_f1": bil_f1, "passes_gate": bil_f1 > 0.20},
        "_excluded": {
            "model": "DNABERT-2",
            "reason": ("Custom ALiBi implementation in the DNABERT-2 remote "
                       "code is incompatible with PyTorch 2.14 on CPU. Tested "
                       "with low_cpu_mem_usage=False, eager attention, pinned "
                       "config, and a manual patch to bert_layers.py; the "
                       "constructor still fails with 'Cannot copy out of "
                       "meta tensor'."),
            "impact": ("Cross-architecture comparison uses CNN and BiLSTM. "
                       "Eight attributions per sequence (2 models x 4 methods) "
                       "replace the planned twelve."),
        },
    }
    out = RES_DIR / "train_metrics.json"
    json.dump(metrics, open(out, "w"), indent=2)
    print(f"wrote {out}")
    for k, v in metrics.items():
        if k.startswith("_"): continue
        print(f"  {k:10s}  val_macro_f1 = {v['val_macro_f1']:.4f}  gate = {v['passes_gate']}")


if __name__ == "__main__":
    main()
