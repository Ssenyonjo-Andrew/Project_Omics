"""Precompute per-sequence results and gene spans for the Streamlit prototype.
Writes results/prototype_data.json -- the app is a pure reader."""
import json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio import SeqIO

CFG = yaml.safe_load(open("config.yaml"))
FAMILIES = CFG["models"]["families"]; FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N = len(FAMILIES)
CKPT = Path(CFG["training"]["checkpoint_dir"])
RES  = Path(CFG["training"]["results_dir"])
DEV = "cpu"
BIN_BP = 100


def windows(seq, wlen, stride):
    if len(seq) < wlen: yield seq + "N" * (wlen - len(seq)); return
    for i in range(0, len(seq) - wlen + 1, stride): yield seq[i:i + wlen]
    if (len(seq) - wlen) % stride: yield seq[-wlen:]


DNA = {"A": 0, "C": 1, "G": 2, "T": 3}
def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in DNA: x[DNA[c], i] = 1.0
    return x


def kmers(s, k):
    ids = []
    for i in range(0, len(s) - k + 1, k):
        idx, ok = 0, True
        for c in s[i:i + k]:
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
        self.fc = nn.Linear(f2, N)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


class BiLSTM(nn.Module):
    def __init__(self, v, e, h, dr):
        super().__init__()
        self.emb = nn.Embedding(v, e, padding_idx=0)
        self.lstm = nn.LSTM(e, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dr); self.fc = nn.Linear(h * 2, N)
    def forward(self, x):
        out, _ = self.lstm(self.emb(x)); return self.fc(self.drop(out.mean(1)))


@torch.no_grad()
def predict(model, seqs, window, stride, k=None, is_lstm=False, batch=32):
    ws, owner = [], []
    for i, s in enumerate(seqs):
        for w in windows(s, window, stride):
            ws.append(kmers(w, k) if is_lstm else onehot(w))
            owner.append(i)
    owner = np.array(owner)
    if is_lstm:
        L = max(len(t) for t in ws)
        X = np.zeros((len(ws), L), dtype=np.int64)
        for i, t in enumerate(ws): X[i, :len(t)] = t
        X = torch.from_numpy(X)
    else:
        X = torch.from_numpy(np.stack(ws))
    probs = []
    for j in range(0, len(X), batch):
        probs.append(F.softmax(model(X[j:j+batch]), -1).numpy())
    probs = np.concatenate(probs)
    out = np.zeros((len(seqs), N))
    for i in range(len(seqs)):
        out[i] = probs[owner == i].mean(0)
    return out


def get_gene_spans(acc):
    gb = Path(f"data/raw/genbank/{acc}.gb")
    if not gb.exists() or gb.stat().st_size == 0:
        return {}
    genes = defaultdict(list)
    try:
        for rec in SeqIO.parse(str(gb), "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS", "mat_peptide", "gene"): continue
                name = None
                for key in ("gene", "product", "standard_name"):
                    if key in feat.qualifiers:
                        name = feat.qualifiers[key][0]; break
                if not name: continue
                name = name.upper().strip()
                for suf in (" PROTEIN", " POLYPROTEIN", " (PP)"):
                    if name.endswith(suf):
                        name = name[:-len(suf)].strip()
                try:
                    genes[name].append([int(feat.location.start), int(feat.location.end)])
                except Exception:
                    for part in feat.location.parts:
                        genes[name].append([int(part.start), int(part.end)])
    except Exception:
        return {}
    return dict(genes)


def main():
    meta = json.load(open(RES / "attr_metadata.json"))
    accs = meta["sample_accessions"]
    master = pd.read_csv("data/processed/master_final.csv")
    master_by_acc = {r["accession"]: r for _, r in master.iterrows()}

    # load sequences
    seq_cache = {}
    seqs = {}
    families = {}
    for acc in accs:
        row = master_by_acc.get(acc)
        if row is None: continue
        p = row["sequence_file_path"]
        if p not in seq_cache:
            seq_cache[p] = {rec.id.split(".")[0]: str(rec.seq) for rec in SeqIO.parse(p, "fasta")}
        s = seq_cache[p].get(acc.split(".")[0])
        if s is None: continue
        seqs[acc] = s.upper()
        families[acc] = row["family"]

    print(f"loaded {len(seqs)} sequences")

    # load models
    c = CFG["models"]["cnn"]; b = CFG["models"]["bilstm"]
    cnn = CNN(c["filters"][0], c["filters"][1], c["kernel"], c["dropout"]).to(DEV)
    cnn.load_state_dict(torch.load(CKPT/"cnn.pt", map_location=DEV)); cnn.eval()
    bil = BiLSTM(b["vocab"], b["embed"], b["hidden"], b["dropout"]).to(DEV)
    bil.load_state_dict(torch.load(CKPT/"bilstm.pt", map_location=DEV)); bil.eval()

    seq_list = list(seqs.keys())
    seq_strings = [seqs[a] for a in seq_list]

    print("predicting (cnn)...")
    cnn_probs = predict(cnn, seq_strings, c["window"], c["stride"])
    print("predicting (bilstm)...")
    bil_probs = predict(bil, seq_strings, b["window"], b["stride"], b["kmer"], is_lstm=True)

    # load eval, per_gene, train metrics
    train_m = json.load(open(RES / "train_metrics.json"))
    eval_m  = json.load(open(RES / "eval" / "evaluation.json"))
    per_g   = json.load(open(RES / "eval" / "per_gene.json"))

    # assemble per-sequence dict
    sequences = {}
    for i, acc in enumerate(seq_list):
        cons_p = RES / "consensus" / f"{acc}.npy"
        if not cons_p.exists(): continue
        cons = np.load(cons_p).tolist()
        n_bins = len(cons)

        # method top-5
        top5 = {}
        for model_tag in ("cnn", "bilstm"):
            for method in ("occlusion", "permutation", "ig", "kernelshap"):
                p = RES / "attr" / f"{model_tag}_{method}" / f"{acc}.npy"
                if not p.exists(): continue
                arr = np.abs(np.load(p))
                top5[f"{model_tag}_{method}"] = np.argsort(-arr)[:5].tolist()

        # genes with int coords
        genes = get_gene_spans(acc)
        genes_serializable = {k: v for k, v in genes.items()}

        stab = eval_m["stability"].get(acc, {})
        sequences[acc] = {
            "family": families[acc],
            "length": len(seqs[acc]),
            "n_bins": n_bins,
            "bin_bp": BIN_BP,
            "cnn": {
                "predicted": FAMILIES[int(np.argmax(cnn_probs[i]))],
                "probs": cnn_probs[i].tolist(),
            },
            "bilstm": {
                "predicted": FAMILIES[int(np.argmax(bil_probs[i]))],
                "probs": bil_probs[i].tolist(),
            },
            "consensus": cons,
            "method_top5": top5,
            "genes": genes_serializable,
            "stability": {
                "median": stab.get("median_spearman"),
                "lo": stab.get("lower_95"),
                "hi": stab.get("upper_95"),
            },
        }
        print(f"  {acc} ({families[acc]}) -> {sequences[acc]['cnn']['predicted']} / "
              f"{sequences[acc]['bilstm']['predicted']}")

    # headline numbers
    stabs = [v["median_spearman"] for v in eval_m["stability"].values()]
    headline = {
        "cnn_test_f1": train_m["cnn"]["test"]["macro_f1"],
        "bilstm_test_f1": train_m["bilstm"]["test"]["macro_f1"],
        "chance_3way": 0.3333,
        "stability_median": float(np.median(stabs)),
        "n_sequences": len(sequences),
        "families": ["Coronaviridae", "Filoviridae", "Flaviviridae"],
        "test_n": {"Coronaviridae": 87, "Filoviridae": 87, "Flaviviridae": 87},
    }

    out = {
        "headline": headline,
        "sequences": sequences,
        "per_gene": per_g["summary"],
        "fidelity": eval_m["fidelity"]["per_sequence"],
        "plausibility_cds": eval_m["plausibility_summary_by_family"],
    }
    (RES / "prototype_data.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/prototype_data.json ({len(sequences)} sequences)")


if __name__ == "__main__":
    main()
