"""Phase 4: attribution (4 methods x 2 models) + consensus ranking.

Design choices forced by CPU-only hardware, documented in attr_metadata.json:
  * 10 test sequences per held-out family (20 total)
  * 100 bp attribution bins (consensus resolution)
  * 16-step Integrated Gradients
  * KernelSHAP with 100 sampled coalitions
"""
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio import SeqIO

CFG = yaml.safe_load(open("config.yaml"))
SEED = CFG["project"]["seed"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

FAMILIES = CFG["models"]["families"]
FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N_CLASSES = len(FAMILIES)
CKPT_DIR = Path(CFG["training"]["checkpoint_dir"])
RES_DIR  = Path(CFG["training"]["results_dir"])
ATTR_DIR = RES_DIR / "attr"; ATTR_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cpu"

BIN_BP = 100
IG_STEPS = 16
KS_COALITIONS = 100
SAMPLE_PER_FAMILY = 15


# ---------- data ----------
def load_test_records():
    df = pd.read_csv("data/processed/master_final.csv")
    df = df[df["split"] == "test"].reset_index(drop=True)
    cache, out = {}, []
    for _, r in df.iterrows():
        p = r["sequence_file_path"]
        if p not in cache:
            cache[p] = {rec.id.split(".")[0]: str(rec.seq) for rec in SeqIO.parse(p, "fasta")}
        seq = cache[p].get(r["accession"].split(".")[0])
        if seq:
            out.append({"accession": r["accession"], "sequence": seq.upper(),
                        "family": r["family"]})
    return out


def sample_by_family(records, per_family):
    by_fam = {}
    for r in records:
        by_fam.setdefault(r["family"], []).append(r)
    sampled = []
    for fam, items in by_fam.items():
        rng = random.Random(SEED + (hash(fam) & 0xFFFF))
        rng.shuffle(items)
        sampled.extend(items[:per_family])
    return sampled


def load_train_background():
    """One training sequence per family, for KernelSHAP reference."""
    df = pd.read_csv("data/processed/master_final.csv")
    df = df[df["split"] == "train"]
    cache, bg = {}, {}
    for _, r in df.iterrows():
        if r["family"] in bg: continue
        p = r["sequence_file_path"]
        if p not in cache:
            cache[p] = {rec.id.split(".")[0]: str(rec.seq) for rec in SeqIO.parse(p, "fasta")}
        seq = cache[p].get(r["accession"].split(".")[0])
        if seq: bg[r["family"]] = seq.upper()
    return bg


# ---------- encodings ----------
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


# ---------- models ----------
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
def predict_cnn(model, seqs, window, stride, batch=64):
    model.eval()
    all_windows, owner = [], []
    for i, s in enumerate(seqs):
        for w in windows(s, window, stride):
            all_windows.append(onehot(w)); owner.append(i)
    X = np.stack(all_windows); owner = np.array(owner)
    probs = []
    for j in range(0, len(X), batch):
        probs.append(F.softmax(model(torch.from_numpy(X[j:j+batch])), -1).numpy())
    probs = np.concatenate(probs)
    out = np.zeros((len(seqs), N_CLASSES))
    for i in range(len(seqs)):
        out[i] = probs[owner == i].mean(0)
    return out


@torch.no_grad()
def predict_bilstm(model, seqs, window, stride, k, batch=64):
    model.eval()
    all_windows, owner = [], []
    for i, s in enumerate(seqs):
        for w in windows(s, window, stride):
            all_windows.append(kmers(w, k)); owner.append(i)
    L = max(len(t) for t in all_windows)
    X = np.zeros((len(all_windows), L), dtype=np.int64)
    for i, t in enumerate(all_windows): X[i, :len(t)] = t
    owner = np.array(owner)
    probs = []
    for j in range(0, len(X), batch):
        probs.append(F.softmax(model(torch.from_numpy(X[j:j+batch])), -1).numpy())
    probs = np.concatenate(probs)
    out = np.zeros((len(seqs), N_CLASSES))
    for i in range(len(seqs)):
        out[i] = probs[owner == i].mean(0)
    return out


# ---------- attribution methods ----------
def occlude(seq, bin_bp=BIN_BP):
    n = (len(seq) + bin_bp - 1) // bin_bp
    return [seq[:b*bin_bp] + "N"*min(bin_bp, len(seq)-b*bin_bp) + seq[(b+1)*bin_bp:]
            for b in range(n)], n


def permute(seq, bin_bp=BIN_BP, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    n = (len(seq) + bin_bp - 1) // bin_bp
    out = []
    for b in range(n):
        s, e = b*bin_bp, min(b*bin_bp+bin_bp, len(seq))
        chunk = list(seq[s:e]); rng.shuffle(chunk)
        out.append(seq[:s] + "".join(chunk) + seq[e:])
    return out, n


def ig_cnn(model, seq, fam_idx, window, stride, steps):
    L = len(seq); n_bins = (L + BIN_BP - 1) // BIN_BP
    attr = np.zeros(n_bins)
    starts = list(range(0, max(L-window, 0)+1, stride))
    if not starts or starts[-1] != L-window: starts.append(max(L-window, 0))
    for s in starts:
        win = seq[s:s+window]
        if len(win) < window: win += "N"*(window-len(win))
        x = torch.from_numpy(onehot(win))
        b = torch.full_like(x, 0.25)
        d = x - b
        alphas = torch.linspace(0, 1, steps).reshape(-1, 1, 1)
        interp = (b + alphas*d).detach().requires_grad_(True)
        out = model(interp)
        grads = torch.autograd.grad(out[:, fam_idx].sum(), interp)[0]
        per_pos = (grads.mean(0) * d).sum(0).abs().numpy()
        for p, v in enumerate(per_pos):
            if s+p >= L: break
            attr[(s+p)//BIN_BP] += v
    return attr


def ig_bilstm(model, seq, fam_idx, window, stride, k, steps):
    L = len(seq); n_bins = (L + BIN_BP - 1) // BIN_BP
    attr = np.zeros(n_bins)
    starts = list(range(0, max(L-window, 0)+1, stride))
    if not starts or starts[-1] != L-window: starts.append(max(L-window, 0))
    for s in starts:
        win = seq[s:s+window]
        if len(win) < window: win += "N"*(window-len(win))
        tok = torch.tensor([kmers(win, k)], dtype=torch.long)
        emb = model.emb(tok).detach()                       # (1, T, E)
        base = torch.zeros_like(emb)
        d = emb - base
        alphas = torch.linspace(0, 1, steps).reshape(-1, 1, 1)
        interp = (base + alphas*d).detach().requires_grad_(True)
        lstm_out, _ = model.lstm(interp)
        logits = model.fc(model.drop(lstm_out.mean(1)))
        grads = torch.autograd.grad(logits[:, fam_idx].sum(), interp)[0]
        per_tok = (grads.mean(0) * d[0]).sum(-1).abs().numpy()  # (T,)
        for t, v in enumerate(per_tok):
            for p in range(t*k, (t+1)*k):
                if s+p >= L: break
                attr[(s+p)//BIN_BP] += v / k
    return attr


def kernelshap_approx(predict_fn, seq, bg, fam_idx, n_bins, n_coal=KS_COALITIONS, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    from math import comb
    seqs, Z = [], []
    for _ in range(n_coal):
        k = int(rng.integers(1, n_bins))
        z = np.zeros(n_bins, dtype=int)
        z[rng.choice(n_bins, size=k, replace=False)] = 1
        built = ""
        for b in range(n_bins):
            s, e = b*BIN_BP, min(b*BIN_BP+BIN_BP, len(seq))
            if z[b]:
                built += seq[s:e]
            else:
                bs = b*BIN_BP
                if bs < len(bg):
                    be = min(bs+BIN_BP, len(bg)); built += bg[bs:be]
                    built += "N" * max(0, (e-s)-(be-bs))
                else:
                    built += "N" * (e-s)
        seqs.append(built); Z.append(z)
    probs = predict_fn(seqs)
    y = probs[:, fam_idx]
    Z = np.stack(Z)
    A = np.hstack([np.ones((len(Z), 1)), Z])
    w = np.array([(n_bins-1)/(comb(n_bins, int(z.sum())) * int(z.sum()) * (n_bins-int(z.sum())))
                  for z in Z])
    AtW = A.T * w
    AtA = AtW @ A + 1e-6 * np.eye(n_bins+1)
    Aty = AtW @ y
    try:
        phi = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
    return phi[1:]


# ---------- consensus ----------
def rank1(x):
    """Higher value -> rank 1."""
    order = np.argsort(-np.abs(x))
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x)+1)
    return r


def main():
    print("loading models...")
    c = CFG["models"]["cnn"]; b = CFG["models"]["bilstm"]
    cnn = CNN(c["filters"][0], c["filters"][1], c["kernel"], c["dropout"]).to(DEV)
    cnn.load_state_dict(torch.load(CKPT_DIR / "cnn.pt", map_location=DEV))
    cnn.eval()
    bil = BiLSTM(b["vocab"], b["embed"], b["hidden"], b["dropout"]).to(DEV)
    bil.load_state_dict(torch.load(CKPT_DIR / "bilstm.pt", map_location=DEV))
    bil.eval()

    print("loading data...")
    test = load_test_records()
    sample = sample_by_family(test, SAMPLE_PER_FAMILY)
    print(f"  test: {len(test)} seqs, sampled: {len(sample)}")
    bg = load_train_background()
    print(f"  backgrounds per family: {list(bg.keys())}")

    for i, r in enumerate(sample):
        acc, seq, fam = r["accession"], r["sequence"], r["family"]
        fidx = FAM2IDX[fam]; nb = (len(seq) + BIN_BP - 1) // BIN_BP
        print(f"\n[{i+1}/{len(sample)}] {acc} {fam} L={len(seq)} bins={nb}")

        def save(key, a):
            d = ATTR_DIR / key; d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{acc}.npy", a)

        # CNN
        if not (ATTR_DIR/"cnn_occlusion"/f"{acc}.npy").exists():
            print("  cnn occlusion")
            ms, _ = occlude(seq)
            p0 = predict_cnn(cnn, [seq], c["window"], c["stride"])[0, fidx]
            pm = predict_cnn(cnn, ms, c["window"], c["stride"])[:, fidx]
            save("cnn_occlusion", p0 - pm)
        if not (ATTR_DIR/"cnn_permutation"/f"{acc}.npy").exists():
            print("  cnn permutation")
            ms, _ = permute(seq, rng=np.random.default_rng(SEED+i))
            p0 = predict_cnn(cnn, [seq], c["window"], c["stride"])[0, fidx]
            pm = predict_cnn(cnn, ms, c["window"], c["stride"])[:, fidx]
            save("cnn_permutation", p0 - pm)
        if not (ATTR_DIR/"cnn_ig"/f"{acc}.npy").exists():
            print("  cnn IG")
            save("cnn_ig", ig_cnn(cnn, seq, fidx, c["window"], c["stride"], IG_STEPS))
        if not (ATTR_DIR/"cnn_kernelshap"/f"{acc}.npy").exists():
            print("  cnn KernelSHAP")
            fn = lambda sqs: predict_cnn(cnn, sqs, c["window"], c["stride"])
            save("cnn_kernelshap",
                 kernelshap_approx(fn, seq, bg.get(fam, seq), fidx, nb,
                                   rng=np.random.default_rng(SEED+i+100)))

        # BiLSTM
        if not (ATTR_DIR/"bilstm_occlusion"/f"{acc}.npy").exists():
            print("  bilstm occlusion")
            ms, _ = occlude(seq)
            p0 = predict_bilstm(bil, [seq], b["window"], b["stride"], b["kmer"])[0, fidx]
            pm = predict_bilstm(bil, ms, b["window"], b["stride"], b["kmer"])[:, fidx]
            save("bilstm_occlusion", p0 - pm)
        if not (ATTR_DIR/"bilstm_permutation"/f"{acc}.npy").exists():
            print("  bilstm permutation")
            ms, _ = permute(seq, rng=np.random.default_rng(SEED+i))
            p0 = predict_bilstm(bil, [seq], b["window"], b["stride"], b["kmer"])[0, fidx]
            pm = predict_bilstm(bil, ms, b["window"], b["stride"], b["kmer"])[:, fidx]
            save("bilstm_permutation", p0 - pm)
        if not (ATTR_DIR/"bilstm_ig"/f"{acc}.npy").exists():
            print("  bilstm IG")
            save("bilstm_ig",
                 ig_bilstm(bil, seq, fidx, b["window"], b["stride"], b["kmer"], IG_STEPS))
        if not (ATTR_DIR/"bilstm_kernelshap"/f"{acc}.npy").exists():
            print("  bilstm KernelSHAP")
            fn = lambda sqs: predict_bilstm(bil, sqs, b["window"], b["stride"], b["kmer"])
            save("bilstm_kernelshap",
                 kernelshap_approx(fn, seq, bg.get(fam, seq), fidx, nb,
                                   rng=np.random.default_rng(SEED+i+100)))

    print("\ncomputing consensus...")
    cons_dir = RES_DIR / "consensus"; cons_dir.mkdir(parents=True, exist_ok=True)
    for r in sample:
        acc = r["accession"]
        mats = []
        for m in ["cnn", "bilstm"]:
            for meth in ["occlusion", "permutation", "ig", "kernelshap"]:
                p = ATTR_DIR / f"{m}_{meth}" / f"{acc}.npy"
                if p.exists(): mats.append(np.load(p))
        if len(mats) < 4:
            print(f"  {acc}: only {len(mats)} attributions, skipping")
            continue
        ranks = np.stack([rank1(x) for x in mats])
        n = ranks.shape[1]
        rp = np.exp((np.log(ranks / n)).sum(0) / ranks.shape[0])
        np.save(cons_dir / f"{acc}.npy", rp)

    meta = {
        "sample_accessions": [r["accession"] for r in sample],
        "families": {r["accession"]: r["family"] for r in sample},
        "bin_bp": BIN_BP,
        "methods": ["occlusion", "permutation", "ig", "kernelshap"],
        "models": ["cnn", "bilstm"],
        "ig_steps": IG_STEPS,
        "kernelshap_coalitions": KS_COALITIONS,
        "note": ("CPU-constrained design: 10 sequences per held-out family, "
                 "100 bp bins, 16-step IG, 100 KernelSHAP coalitions."),
    }
    json.dump(meta, open(RES_DIR / "attr_metadata.json", "w"), indent=2)
    print(f"\nDone. Attributions: {ATTR_DIR}")
    print(f"Consensus: {cons_dir}")
    print(f"Metadata: {RES_DIR / 'attr_metadata.json'}")


if __name__ == "__main__":
    main()
