
"""Phase 7: compute 8 attributions per test sequence (4 methods x 2 models)."""
import json, random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from Bio import SeqIO

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

VIRUSES = ["dengue", "chikungunya", "hcv", "norovirus"]
VIR_TO_IDX = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
BIN_BP = 100
WINDOW = 3000
STRIDE = 1500
IG_STEPS = 16
KS_COALITIONS = 50
N_PER_VIRUS = 8

CKPT = Path("models")
OUT = Path("results/attr_v2")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cpu"


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


def load_models():
    cnn = CNN(); cnn.load_state_dict(torch.load(CKPT / "cnn_virus.pt", map_location=DEV))
    cnn.eval()
    bil = BiLSTM(); bil.load_state_dict(torch.load(CKPT / "bilstm_virus.pt", map_location=DEV))
    bil.eval()
    return cnn, bil


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


@torch.no_grad()
def predict_batch(model, seqs, is_lstm):
    out = []
    for s in seqs:
        ws = list(windows(s, WINDOW, STRIDE))
        if is_lstm:
            toks = [kmers(w) for w in ws]
            L = max(len(t) for t in toks)
            X = torch.zeros(len(toks), L, dtype=torch.long)
            for i, t in enumerate(toks): X[i, :len(t)] = torch.tensor(t)
            p = F.softmax(model(X), -1).mean(0)
        else:
            X = torch.stack([torch.from_numpy(onehot(w)) for w in ws])
            p = F.softmax(model(X), -1).mean(0)
        out.append(p.numpy())
    return np.stack(out)


def bins_from_seq(seq):
    n = (len(seq) + BIN_BP - 1) // BIN_BP
    return n


def occlude(seq, cls, model, is_lstm):
    n = bins_from_seq(seq)
    variants = []
    for b in range(n):
        s, e = b * BIN_BP, min((b + 1) * BIN_BP, len(seq))
        variants.append(seq[:s] + "N" * (e - s) + seq[e:])
    base = predict_batch(model, [seq], is_lstm)[0, cls]
    mut = predict_batch(model, variants, is_lstm)[:, cls]
    return base - mut


def permute_bins(seq, cls, model, is_lstm, rng):
    n = bins_from_seq(seq)
    variants = []
    for b in range(n):
        s, e = b * BIN_BP, min((b + 1) * BIN_BP, len(seq))
        chunk = list(seq[s:e]); rng.shuffle(chunk)
        variants.append(seq[:s] + "".join(chunk) + seq[e:])
    base = predict_batch(model, [seq], is_lstm)[0, cls]
    mut = predict_batch(model, variants, is_lstm)[:, cls]
    return base - mut


def integrated_gradients(seq, cls, model, is_lstm):
    n_bins = bins_from_seq(seq)
    attr = np.zeros(n_bins)
    for wstart in range(0, max(len(seq) - WINDOW, 0) + 1, STRIDE):
        win = seq[wstart:wstart + WINDOW]
        if len(win) < WINDOW: win += "N" * (WINDOW - len(win))
        if is_lstm:
            tok = torch.tensor([kmers(win)], dtype=torch.long)
            emb = model.emb(tok).detach()
            base = torch.zeros_like(emb)
            d = emb - base
            steps = [base + (i / IG_STEPS) * d for i in range(IG_STEPS + 1)]
            grads = []
            for st in steps:
                st = st.detach().requires_grad_(True)
                o, _ = model.lstm(st)
                logits = model.fc(model.drop(o.mean(1)))
                g = torch.autograd.grad(logits[0, cls], st)[0]
                grads.append(g)
            avg = torch.stack(grads).mean(0)
            ig = (avg * d[0]).sum(-1).detach().numpy().flatten()
            for t, v in enumerate(ig):
                pos = wstart + t * 6
                if pos < len(seq):
                    attr[pos // BIN_BP] += abs(v) / 6.0
        else:
            x = torch.from_numpy(onehot(win))
            base = torch.full_like(x, 0.25)
            d = x - base
            steps = [base + (i / IG_STEPS) * d for i in range(IG_STEPS + 1)]
            grads = []
            for st in steps:
                st = st.detach().requires_grad_(True)
                logits = model(st.unsqueeze(0))
                g = torch.autograd.grad(logits[0, cls], st)[0]
                grads.append(g)
            avg = torch.stack(grads).mean(0)
            ig = (avg * d).sum(0).detach().numpy()
            for p, v in enumerate(ig):
                if wstart + p < len(seq):
                    attr[(wstart + p) // BIN_BP] += abs(v)
    return attr


def kernelshap(seq, cls, model, is_lstm, rng):
    n = bins_from_seq(seq)
    variants = []
    for _ in range(KS_COALITIONS):
        z = rng.integers(0, 2, size=n)
        variant = "".join(
            (seq[b * BIN_BP: min((b + 1) * BIN_BP, len(seq))] if z[b]
             else "N" * (min((b + 1) * BIN_BP, len(seq)) - b * BIN_BP))
            for b in range(n)
        )
        variants.append((z, variant))
    probs = predict_batch(model, [v[1] for v in variants], is_lstm)[:, cls]
    Z = np.stack([v[0] for v in variants]).astype(np.float32)
    Z = np.hstack([np.ones((len(Z), 1)), Z])
    try:
        coef = np.linalg.lstsq(Z, probs, rcond=None)[0]
        return coef[1:]
    except np.linalg.LinAlgError:
        return np.zeros(n)


def load_test_samples():
    samples = {}
    for v in VIRUSES:
        fasta = Path(f"data/processed/split/{v}_test.fasta")
        recs = list(SeqIO.parse(fasta, "fasta"))
        rng = random.Random(SEED)
        if len(recs) > N_PER_VIRUS:
            recs = rng.sample(recs, N_PER_VIRUS)
        samples[v] = [(r.id, str(r.seq).upper()) for r in recs]
    return samples


def main():
    print("loading models...")
    cnn, bil = load_models()

    print("loading samples...")
    samples = load_test_samples()
    for v in VIRUSES:
        print(f"  {v}: {len(samples[v])}")

    predictions = {}

    for virus in VIRUSES:
        for acc, seq in samples[virus]:
            print(f"\n=== {virus} / {acc} ({len(seq)} bp) ===")
            preds = {"true_virus": virus, "sequence_length": len(seq)}

            # CNN
            p_cnn = predict_batch(cnn, [seq], False)[0]
            preds["cnn_pred"] = VIRUSES[int(np.argmax(p_cnn))]
            preds["cnn_probs"] = p_cnn.tolist()
            cls_cnn = int(np.argmax(p_cnn))

            # BiLSTM
            p_bil = predict_batch(bil, [seq], True)[0]
            preds["bilstm_pred"] = VIRUSES[int(np.argmax(p_bil))]
            preds["bilstm_probs"] = p_bil.tolist()
            cls_bil = int(np.argmax(p_bil))

            print(f"  cnn -> {preds['cnn_pred']} ({p_cnn.max():.3f})  "
                  f"bilstm -> {preds['bilstm_pred']} ({p_bil.max():.3f})")

            rng = np.random.default_rng(SEED)

            for tag, model, is_lstm, cls in [
                ("cnn", cnn, False, cls_cnn),
                ("bilstm", bil, True, cls_bil),
            ]:
                for method in ["occlusion", "permutation", "ig", "kernelshap"]:
                    path = OUT / tag / method / f"{acc}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        print(f"  {tag}/{method}: cached")
                        continue
                    print(f"  {tag}/{method}...")
                    if method == "occlusion":
                        attr = occlude(seq, cls, model, is_lstm)
                    elif method == "permutation":
                        attr = permute_bins(seq, cls, model, is_lstm, rng)
                    elif method == "ig":
                        attr = integrated_gradients(seq, cls, model, is_lstm)
                    elif method == "kernelshap":
                        attr = kernelshap(seq, cls, model, is_lstm, rng)
                    np.save(path, np.abs(attr))
            predictions[acc] = preds

    json.dump(predictions, open(OUT / "predictions.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'predictions.json'}")
    print(f"done: {len(predictions)} sequences processed")


if __name__ == "__main__":
    main()
