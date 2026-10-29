"""Phase 5: stability, fidelity, plausibility evaluation."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr, hypergeom
from Bio import SeqIO

CFG = yaml.safe_load(open("config.yaml"))
FAMILIES = CFG["models"]["families"]
FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N_CLASSES = len(FAMILIES)
CKPT_DIR = Path(CFG["training"]["checkpoint_dir"])
RES_DIR  = Path(CFG["training"]["results_dir"])
ATTR_DIR = RES_DIR / "attr"
CONS_DIR = RES_DIR / "consensus"
EVAL_DIR = RES_DIR / "eval"; EVAL_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cpu"

BIN_BP = 100
N_BOOTSTRAP = 500
DEL_LEVELS = [0.01, 0.05, 0.10, 0.20]
PLAUS_TOP_FRAC = 0.05

WITH_RETRAIN = "--with-retrain" in sys.argv


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
    ws, owner = [], []
    for i, s in enumerate(seqs):
        for w in windows(s, window, stride):
            ws.append(onehot(w)); owner.append(i)
    X = np.stack(ws); owner = np.array(owner)
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
    ws, owner = [], []
    for i, s in enumerate(seqs):
        for w in windows(s, window, stride):
            ws.append(kmers(w, k)); owner.append(i)
    L = max(len(t) for t in ws)
    X = np.zeros((len(ws), L), dtype=np.int64)
    for i, t in enumerate(ws): X[i, :len(t)] = t
    owner = np.array(owner)
    probs = []
    for j in range(0, len(X), batch):
        probs.append(F.softmax(model(torch.from_numpy(X[j:j+batch])), -1).numpy())
    probs = np.concatenate(probs)
    out = np.zeros((len(seqs), N_CLASSES))
    for i in range(len(seqs)):
        out[i] = probs[owner == i].mean(0)
    return out


# ---------- stability ----------
def stability():
    meta = json.load(open(RES_DIR / "attr_metadata.json"))
    accs = meta["sample_accessions"]
    per_acc = {}
    rng = np.random.default_rng(42)

    for acc in accs:
        mats = []
        for m in ["cnn", "bilstm"]:
            for meth in ["occlusion", "permutation", "ig", "kernelshap"]:
                p = ATTR_DIR / f"{m}_{meth}" / f"{acc}.npy"
                if p.exists(): mats.append(np.load(p))
        if len(mats) < 4: continue
        M = np.stack([np.abs(x) for x in mats])  # (8, bins)
        n_bins = M.shape[1]
        rho = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, M.shape[0], size=M.shape[0])
            sample = M[idx]
            A = sample[:4].mean(0); B = sample[4:].mean(0)
            if A.std() < 1e-9 or B.std() < 1e-9:
                continue
            r = spearmanr(A, B).correlation
            if not np.isnan(r): rho.append(r)
        per_acc[acc] = {"median_spearman": float(np.median(rho)),
                        "lower_95": float(np.percentile(rho, 2.5)),
                        "upper_95": float(np.percentile(rho, 97.5)),
                        "n_boot": len(rho)}
    return per_acc


# ---------- fidelity ----------
def fidelity(cnn, bil, meta, sample_records):
    """Deletion/insertion curves for the consensus ranking."""
    results = {"per_sequence": {}}
    c = CFG["models"]["cnn"]; b = CFG["models"]["bilstm"]

    for r in sample_records:
        acc, seq, fam = r["accession"], r["sequence"], r["family"]
        fidx = FAM2IDX[fam]
        cons_path = CONS_DIR / f"{acc}.npy"
        if not cons_path.exists(): continue
        cons = np.load(cons_path)
        n_bins = len(cons); L = len(seq)
        # rank bins: high consensus rank (low value) = most important
        order = np.argsort(cons)

        # baseline probability
        p0_cnn = predict_cnn(cnn, [seq], c["window"], c["stride"])[0, fidx]
        p0_bil = predict_bilstm(bil, [seq], b["window"], b["stride"], b["kmer"])[0, fidx]

        rec = {"family": fam, "L": L,
               "p0_cnn": float(p0_cnn), "p0_bilstm": float(p0_bil),
               "deletion": {}, "insertion": {}}

        for frac in DEL_LEVELS:
            n_del = max(1, int(frac * n_bins))
            mask = set(order[:n_del])
            deleted_seq = "".join("N" if (i // BIN_BP) in mask else ch
                                  for i, ch in enumerate(seq))
            pd_cnn = predict_cnn(cnn, [deleted_seq], c["window"], c["stride"])[0, fidx]
            pd_bil = predict_bilstm(bil, [deleted_seq], b["window"], b["stride"], b["kmer"])[0, fidx]

            # random control
            rand = np.random.default_rng(42).permutation(n_bins)[:n_del]
            rmask = set(rand.tolist())
            rand_seq = "".join("N" if (i // BIN_BP) in rmask else ch
                               for i, ch in enumerate(seq))
            pr_cnn = predict_cnn(cnn, [rand_seq], c["window"], c["stride"])[0, fidx]
            pr_bil = predict_bilstm(bil, [rand_seq], b["window"], b["stride"], b["kmer"])[0, fidx]

            rec["deletion"][str(frac)] = {
                "drop_cnn":  float(p0_cnn - pd_cnn),
                "drop_bilstm": float(p0_bil - pd_bil),
                "rand_drop_cnn": float(p0_cnn - pr_cnn),
                "rand_drop_bilstm": float(p0_bil - pr_bil),
            }
        results["per_sequence"][acc] = rec
    return results


# ---------- plausibility ----------
def fetch_cds(accession):
    """Fetch CDS spans from NCBI GenBank."""
    import subprocess, tempfile
    from Bio import SeqIO
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".gb", delete=False) as tf:
        path = tf.name
    try:
        r = subprocess.run(["efetch","-db","nuccore","-id",accession,"-format","gb"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not r.stdout.strip(): return []
        open(path,"w").write(r.stdout)
        cds=[]
        for rec in SeqIO.parse(path,"genbank"):
            for feat in rec.features:
                if feat.type != "CDS": continue
                try: cds.append((int(feat.location.start), int(feat.location.end)))
                except Exception:
                    try:
                        for part in feat.location.parts:
                            cds.append((int(part.start), int(part.end)))
                    except Exception: continue
        return cds
    finally:
        Path(path).unlink(missing_ok=True)


def plausibility(meta, sample_records):
    results = {"per_sequence": {}}
    for r in sample_records:
        acc, seq, fam = r["accession"], r["sequence"], r["family"]
        cons_path = CONS_DIR / f"{acc}.npy"
        if not cons_path.exists(): continue
        cons = np.load(cons_path)
        n_bins = len(cons)
        print(f"  {acc}: fetching CDS...")
        try:
            cds = fetch_cds(acc)
        except Exception as e:
            print(f"    efetch failed: {e}"); continue
        if not cds:
            print(f"    no CDS annotations"); continue
        # annotate each bin: is any position inside a CDS?
        bin_annot = np.zeros(n_bins, dtype=bool)
        for s, e in cds:
            start_bin = max(0, s // BIN_BP)
            end_bin   = min(n_bins, (e - 1) // BIN_BP + 1)
            bin_annot[start_bin:end_bin] = True
        n_annot = int(bin_annot.sum())
        if n_annot == 0: continue
        # top 5% of consensus
        k = max(1, int(PLAUS_TOP_FRAC * n_bins))
        top = np.argsort(cons)[:k]
        k_annot = int(bin_annot[top].sum())
        # hypergeometric: P(X >= k_annot), X ~ Hypergeom(n_bins, n_annot, k)
        p_val = float(hypergeom.sf(k_annot - 1, n_bins, n_annot, k))
        obs_exp = (k_annot / k) / (n_annot / n_bins) if n_annot > 0 else 0.0
        results["per_sequence"][acc] = {
            "family": fam, "n_bins": n_bins, "n_annotated_bins": n_annot,
            "top_k": k, "top_k_annotated": k_annot,
            "observed_expected_ratio": float(obs_exp),
            "hypergeom_p": p_val,
        }
        print(f"    bins={n_bins}, annotated={n_annot}, top_k={k}, "
              f"top_k_annotated={k_annot}, O/E={obs_exp:.2f}, p={p_val:.4g}")
    return results


def aggregate(plaus):
    by_fam = {}
    for acc, r in plaus["per_sequence"].items():
        by_fam.setdefault(r["family"], []).append(r["observed_expected_ratio"])
    summary = {}
    for fam, xs in by_fam.items():
        xs = np.array(xs)
        summary[fam] = {
            "n": len(xs),
            "median_oe": float(np.median(xs)),
            "mean_oe":   float(np.mean(xs)),
            "min_oe":    float(np.min(xs)),
            "max_oe":    float(np.max(xs)),
        }
    return summary


def main():
    print("=== stability (bootstrap) ===")
    stab = stability()
    print(f"  {len(stab)} sequences evaluated")
    if stab:
        meds = [v["median_spearman"] for v in stab.values()]
        print(f"  median Spearman across sequences: {np.median(meds):.3f}")

    print("\n=== fidelity (deletion/insertion) ===")
    c = CFG["models"]["cnn"]; b = CFG["models"]["bilstm"]
    cnn = CNN(c["filters"][0], c["filters"][1], c["kernel"], c["dropout"]).to(DEV)
    cnn.load_state_dict(torch.load(CKPT_DIR/"cnn.pt", map_location=DEV)); cnn.eval()
    bil = BiLSTM(b["vocab"], b["embed"], b["hidden"], b["dropout"]).to(DEV)
    bil.load_state_dict(torch.load(CKPT_DIR/"bilstm.pt", map_location=DEV)); bil.eval()

    meta = json.load(open(RES_DIR / "attr_metadata.json"))
    df = pd.read_csv("data/processed/master_final.csv")
    df = df[df["split"] == "test"]
    cache, sample_records = {}, []
    for _, r in df.iterrows():
        if r["accession"] not in meta["sample_accessions"]: continue
        p = r["sequence_file_path"]
        if p not in cache:
            cache[p] = {rec.id.split(".")[0]: str(rec.seq) for rec in SeqIO.parse(p, "fasta")}
        s = cache[p].get(r["accession"].split(".")[0])
        if s:
            sample_records.append({"accession": r["accession"],
                                   "sequence": s.upper(), "family": r["family"]})

    fid = fidelity(cnn, bil, meta, sample_records)
    print(f"  {len(fid['per_sequence'])} sequences evaluated")

    print("\n=== plausibility (NCBI CDS enrichment) ===")
    plaus = plausibility(meta, sample_records)
    plaus_summary = aggregate(plaus)

    out = {
        "stability": stab,
        "fidelity": fid,
        "plausibility": plaus,
        "plausibility_summary_by_family": plaus_summary,
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "del_levels": DEL_LEVELS,
            "plaus_top_frac": PLAUS_TOP_FRAC,
            "bin_bp": BIN_BP,
        },
    }
    json.dump(out, open(EVAL_DIR / "evaluation.json", "w"), indent=2)
    print(f"\nwrote {EVAL_DIR}/evaluation.json")

    print("\n=== per-family plausibility summary ===")
    for fam, r in plaus_summary.items():
        print(f"  {fam:20s}  n={r['n']:2d}  median O/E = {r['median_oe']:.2f}")


if __name__ == "__main__":
    main()
