"""Model loading + inference. Pure functions, no HTTP."""
import subprocess
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO

VIRUSES = ["dengue", "chikungunya", "hcv", "norovirus"]
VIR_TO_IDX = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
WINDOW = 3000
STRIDE = 1500
BIN_BP = 100
DEV = "cpu"

MODELS_DIR = Path("../models")
GB_CACHE = Path("../data/raw/genbank")
GB_CACHE.mkdir(parents=True, exist_ok=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 128, 8, padding="same"); self.b1 = nn.BatchNorm1d(128)
        self.c2 = nn.Conv1d(128, 256, 8, padding="same"); self.b2 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(256, N)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4); self.fc = nn.Linear(256, N)
    def forward(self, x):
        o, _ = self.lstm(self.emb(x))
        return self.fc(self.drop(o.mean(1)))


@lru_cache(maxsize=1)
def _models():
    cnn = CNN()
    cnn.load_state_dict(torch.load(MODELS_DIR / "cnn_virus.pt", map_location=DEV))
    cnn.eval()
    bil = BiLSTM()
    bil.load_state_dict(torch.load(MODELS_DIR / "bilstm_virus.pt", map_location=DEV))
    bil.eval()
    return cnn, bil


def _windows(seq, wlen, stride):
    if len(seq) < wlen:
        yield seq + "N" * (wlen - len(seq)); return
    for i in range(0, len(seq) - wlen + 1, stride):
        yield seq[i:i + wlen]
    if (len(seq) - wlen) % stride:
        yield seq[-wlen:]


DNA = {"A": 0, "C": 1, "G": 2, "T": 3}
def _onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in DNA: x[DNA[c], i] = 1.0
    return x


def _kmers(s, k=6):
    ids = []
    for i in range(0, len(s) - k + 1, k):
        idx, ok = 0, True
        for c in s[i:i+k]:
            if c not in "ACGT": ok = False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]


@torch.no_grad()
def predict(seq: str, model_name: str) -> dict:
    cnn, bil = _models()
    model = cnn if model_name == "cnn" else bil
    is_lstm = model_name == "bilstm"
    ws = list(_windows(seq, WINDOW, STRIDE))
    if is_lstm:
        toks = [_kmers(w) for w in ws]
        L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i, t in enumerate(toks): X[i, :len(t)] = torch.tensor(t)
        p = F.softmax(model(X), -1).mean(0)
    else:
        X = torch.stack([torch.from_numpy(_onehot(w)) for w in ws])
        p = F.softmax(model(X), -1).mean(0)
    p = p.numpy()
    idx = int(np.argmax(p))
    return {"virus": VIRUSES[idx], "probs": p.tolist(), "confidence": float(p[idx])}


@torch.no_grad()
def attribute(seq: str, model_name: str, bin_bp: int = BIN_BP) -> dict:
    """Occlusion attribution. Returns per-bin importance."""
    cnn, bil = _models()
    model = cnn if model_name == "cnn" else bil
    is_lstm = model_name == "bilstm"

    base_pred = predict(seq, model_name)
    cls = VIR_TO_IDX[base_pred["virus"]]

    n_bins = (len(seq) + bin_bp - 1) // bin_bp
    variants = []
    for b in range(n_bins):
        s, e = b * bin_bp, min((b + 1) * bin_bp, len(seq))
        variants.append(seq[:s] + "N" * (e - s) + seq[e:])

    win_per_var = len(list(_windows(seq, WINDOW, STRIDE)))
    all_wins = []
    for v in variants:
        all_wins.extend(list(_windows(v, WINDOW, STRIDE)))

    if is_lstm:
        toks = [_kmers(w) for w in all_wins]
        L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i, t in enumerate(toks): X[i, :len(t)] = torch.tensor(t)
    else:
        X = torch.stack([torch.from_numpy(_onehot(w)) for w in all_wins])

    probs = []
    for i in range(0, len(X), 128):
        p = F.softmax(model(X[i:i+128]), -1).numpy()
        probs.append(p)
    probs = np.concatenate(probs)
    base_prob = base_pred["probs"][cls]

    attr = np.zeros(n_bins)
    for b in range(n_bins):
        win_probs = probs[b * win_per_var:(b + 1) * win_per_var, cls]
        attr[b] = base_prob - win_probs.mean()

    attr_norm = attr / (np.abs(attr).max() + 1e-9)
    return {
        "prediction": base_pred["virus"],
        "probabilities": base_pred["probs"],
        "bin_size_bp": bin_bp,
        "n_bins": n_bins,
        "importance": np.abs(attr).tolist(),
        "importance_signed": attr_norm.tolist(),
    }


def fetch_genes(accession: str) -> dict:
    cache = GB_CACHE / f"{accession}.gb"
    if not cache.exists() or cache.stat().st_size == 0:
        try:
            r = subprocess.run(
                ["efetch", "-db", "nuccore", "-id", accession, "-format", "gb"],
                capture_output=True, text=True, timeout=30)
            if r.returncode != 0 or not r.stdout.strip():
                return {}
            cache.write_text(r.stdout)
        except Exception:
            return {}
    genes: dict = {}
    try:
        for rec in SeqIO.parse(str(cache), "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS", "mat_peptide", "gene"):
                    continue
                name = None
                for key in ("gene", "product", "standard_name"):
                    if key in feat.qualifiers:
                        name = feat.qualifiers[key][0].upper()
                        break
                if not name:
                    continue
                try:
                    genes.setdefault(name, []).append(
                        [int(feat.location.start), int(feat.location.end)])
                except Exception:
                    for part in feat.location.parts:
                        genes.setdefault(name, []).append(
                            [int(part.start), int(part.end)])
    except Exception:
        return {}
    return genes

@torch.no_grad()
def calculate_fidelity(seq: str, model_name: str, attr_dict: dict, drop_frac: float = 0.10) -> dict:
    """Delete the top X% of attributed bins and measure the probability drop."""
    importance = np.array(attr_dict["importance"])
    n_bins = len(importance)
    n_drop = max(1, int(n_bins * drop_frac))
    
    # Get indices of top bins
    top_bins = np.argsort(importance)[-n_drop:]
    
    # Create masked sequence
    bin_bp = attr_dict["bin_size_bp"]
    seq_list = list(seq)
    for b in top_bins:
        s, e = b * bin_bp, min((b + 1) * bin_bp, len(seq))
        for i in range(s, e):
            seq_list[i] = "N"
    masked_seq = "".join(seq_list)
    
    # Get new prediction
    base_pred = predict(seq, model_name)
    masked_pred = predict(masked_seq, model_name)
    
    virus = base_pred["virus"]
    idx = VIR_TO_IDX[virus]
    base_prob = base_pred["probs"][idx]
    masked_prob = masked_pred["probs"][idx]
    
    return {
        "virus": virus,
        "drop_frac": drop_frac,
        "n_bins_deleted": n_drop,
        "base_prob": base_prob,
        "masked_prob": masked_prob,
        "delta": base_prob - masked_prob,
        "new_prediction": masked_pred["virus"]
    }
