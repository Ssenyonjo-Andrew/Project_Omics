
"""Phase 8: stability, fidelity, plausibility."""
import json, random, subprocess, tempfile, re, shutil
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, hypergeom
from Bio import SeqIO

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

VIRUSES = ["dengue", "chikungunya", "hcv", "norovirus"]
VIR_TO_IDX = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
BIN_BP = 100
WINDOW = 3000
STRIDE = 1500
DEL_LEVELS = [0.01, 0.05, 0.10, 0.20]
N_BOOTSTRAP = 500
TOP_FRAC = 0.10
DEV = "cpu"

ATTR = Path("results/attr_v2")
OUT = Path("results/eval_v2"); OUT.mkdir(parents=True, exist_ok=True)
GB_CACHE = Path("data/raw/genbank"); GB_CACHE.mkdir(parents=True, exist_ok=True)

ANTIGENIC = {"dengue": ["E", "ENV"],
             "chikungunya": ["E1"],
             "hcv": ["E2"],
             "norovirus": ["VP1", "CAPSID"]}
CONSERVED = {"dengue": ["NS5"],
             "chikungunya": ["NSP1"],
             "hcv": ["NS5B"],
             "norovirus": ["RDRP"]}


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


def load_models():
    cnn = CNN(); cnn.load_state_dict(torch.load("models/cnn_virus.pt", map_location=DEV)); cnn.eval()
    bil = BiLSTM(); bil.load_state_dict(torch.load("models/bilstm_virus.pt", map_location=DEV)); bil.eval()
    return cnn, bil


def windows(seq, wlen, stride):
    if len(seq) < wlen:
        yield seq + "N" * (wlen - len(seq)); return
    for i in range(0, len(seq) - wlen + 1, stride):
        yield seq[i:i + wlen]
    if (len(seq) - wlen) % stride:
        yield seq[-wlen:]


DNA = {"A":0,"C":1,"G":2,"T":3}
def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in DNA: x[DNA[c], i] = 1.0
    return x


def kmers(s, k=6):
    ids = []
    for i in range(0, len(s) - k + 1, k):
        idx, ok = 0, True
        for c in s[i:i+k]:
            if c not in "ACGT": ok = False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]


@torch.no_grad()
def predict(model, seq, is_lstm):
    ws = list(windows(seq, WINDOW, STRIDE))
    if is_lstm:
        toks = [kmers(w) for w in ws]
        L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i, t in enumerate(toks): X[i, :len(t)] = torch.tensor(t)
        p = F.softmax(model(X), -1).mean(0)
    else:
        X = torch.stack([torch.from_numpy(onehot(w)) for w in ws])
        p = F.softmax(model(X), -1).mean(0)
    return p.numpy()


def rank_low_is_important(x):
    order = np.argsort(-np.abs(x))
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    return r


def consensus(attrs):
    ranks = np.stack([rank_low_is_important(a) for a in attrs])
    return np.exp(np.log(ranks / ranks.shape[1]).mean(0))


def margin(p, cls):
    others = np.delete(p, cls)
    return float(p[cls] - others.max())


def load_sequences():
    seqs = {}
    for v in VIRUSES:
        fasta = Path(f"data/processed/split/{v}_test.fasta")
        for rec in SeqIO.parse(fasta, "fasta"):
            seqs[rec.id] = (str(rec.seq).upper(), v)
    return seqs


def fetch_genes(acc):
    cache = GB_CACHE / f"{acc}.gb"
    if not cache.exists() or cache.stat().st_size == 0:
        try:
            r = subprocess.run(
                ["efetch","-db","nuccore","-id",acc,"-format","gb"],
                capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not r.stdout.strip():
                cache.write_text("")
                return {}
            cache.write_text(r.stdout)
        except Exception:
            return {}
    genes = defaultdict(list)
    try:
        for rec in SeqIO.parse(str(cache), "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS","mat_peptide","gene"): continue
                name = None
                for key in ("gene","product","standard_name"):
                    if key in feat.qualifiers:
                        name = feat.qualifiers[key][0].upper(); break
                if not name: continue
                try:
                    genes[name].append((int(feat.location.start), int(feat.location.end)))
                except Exception:
                    for part in feat.location.parts:
                        genes[name].append((int(part.start), int(part.end)))
    except Exception:
        return {}
    return dict(genes)


def gene_overlaps(gene_spans, bin_start, bin_end):
    for s, e in gene_spans:
        if bin_start < e and bin_end > s:
            return True
    return False


def bins_on_gene(consensus_arr, gene_spans, top_frac, seq_len):
    n_bins = len(consensus_arr)
    k = max(1, int(round(top_frac * n_bins)))
    top = np.argsort(consensus_arr)[:k]
    hits = 0
    for b in top:
        if gene_overlaps(gene_spans, b * BIN_BP, min((b+1)*BIN_BP, seq_len)):
            hits += 1
    return hits, k


def main():
    print("loading models...")
    cnn, bil = load_models()

    print("loading predictions and sequences...")
    preds = json.load(open(ATTR / "predictions.json"))
    seqs = load_sequences()

    accs = list(preds.keys())
    print(f"  {len(accs)} sequences")

    stability_results = {}
    fidelity_results = {}
    plausibility_results = {}

    rng = np.random.default_rng(SEED)

    for i, acc in enumerate(accs):
        seq, true_virus = seqs[acc]
        pred = preds[acc]
        cnn_cls = VIR_TO_IDX[pred["cnn_pred"]]
        bil_cls = VIR_TO_IDX[pred["bilstm_pred"]]
        true_cls = VIR_TO_IDX[true_virus]
        L = len(seq)
        print(f"\n[{i+1}/{len(accs)}] {acc} ({true_virus}, {L} bp)")

        cnn_attrs = []
        bil_attrs = []
        for method in ["occlusion","permutation","ig","kernelshap"]:
            p = ATTR / "cnn" / method / f"{acc}.npy"
            if p.exists(): cnn_attrs.append(np.load(p))
            p = ATTR / "bilstm" / method / f"{acc}.npy"
            if p.exists(): bil_attrs.append(np.load(p))
        if len(cnn_attrs) < 4 or len(bil_attrs) < 4:
            print("  skipping: missing attributions")
            continue

        # --- stability ---
        cnn_cons = consensus(cnn_attrs)
        bil_cons = consensus(bil_attrs)
        if cnn_cons.std() > 1e-9 and bil_cons.std() > 1e-9:
            rho = spearmanr(cnn_cons, bil_cons).correlation
        else:
            rho = np.nan
        stability_results[acc] = {
            "virus": true_virus,
            "spearman": float(rho) if not np.isnan(rho) else None,
            "n_bins": int(len(cnn_cons)),
        }
        print(f"  stability rho = {rho:.3f}" if not np.isnan(rho) else "  stability undefined")

        # --- fidelity ---
        full_cons = consensus(cnn_attrs + bil_attrs)
        n_bins = len(full_cons)
        order = np.argsort(full_cons)

        base_cnn = margin(predict(cnn, seq, False), true_cls)
        base_bil = margin(predict(bil, seq, True), true_cls)

        fid_rec = {"virus": true_virus, "deletion": {}}
        for frac in DEL_LEVELS:
            n_del = max(1, int(round(frac * n_bins)))
            mask = set(order[:n_del].tolist())
            masked_seq = "".join(
                "N" if (j // BIN_BP) in mask else c for j, c in enumerate(seq))
            d_cnn = base_cnn - margin(predict(cnn, masked_seq, False), true_cls)
            d_bil = base_bil - margin(predict(bil, masked_seq, True), true_cls)

            rand_bins = set(rng.choice(n_bins, n_del, replace=False).tolist())
            rand_seq = "".join(
                "N" if (j // BIN_BP) in rand_bins else c for j, c in enumerate(seq))
            r_cnn = base_cnn - margin(predict(cnn, rand_seq, False), true_cls)
            r_bil = base_bil - margin(predict(bil, rand_seq, True), true_cls)

            fid_rec["deletion"][str(frac)] = {
                "cnn": float(d_cnn), "bilstm": float(d_bil),
                "random_cnn": float(r_cnn), "random_bilstm": float(r_bil),
            }
        fidelity_results[acc] = fid_rec
        print(f"  fidelity @20%: cnn +{fid_rec['deletion']['0.2']['cnn']:.4f} "
              f"bilstm +{fid_rec['deletion']['0.2']['bilstm']:.4f}")

        # --- plausibility ---
        genes = fetch_genes(acc)
        if not genes:
            print("  plausibility: no gene annotations")
            continue
        ant_genes = ANTIGENIC[true_virus]
        cons_genes = CONSERVED[true_virus]
        ant_spans = [s for g in genes for name, spans in genes.items()
                     if any(k in name for k in ant_genes) for s in spans]
        con_spans = [s for g in genes for name, spans in genes.items()
                     if any(k in name for k in cons_genes) for s in spans]

        ant_hits, k = bins_on_gene(full_cons, ant_spans, TOP_FRAC, L) if ant_spans else (0, 0)
        con_hits, _ = bins_on_gene(full_cons, con_spans, TOP_FRAC, L) if con_spans else (0, 0)

        # expected from gene coverage
        ant_len = sum(e-s for s, e in ant_spans) if ant_spans else 0
        con_len = sum(e-s for s, e in con_spans) if con_spans else 0
        exp_ant = k * (ant_len / L) if ant_len > 0 else 0
        exp_con = k * (con_len / L) if con_len > 0 else 0

        plausibility_results[acc] = {
            "virus": true_virus,
            "k": int(k),
            "antigenic_genes": list(set(n for n in genes if any(g in n for g in ant_genes))),
            "conserved_genes": list(set(n for n in genes if any(g in n for g in cons_genes))),
            "antigenic_hits": int(ant_hits), "antigenic_expected": float(exp_ant),
            "antigenic_oe": float(ant_hits / exp_ant) if exp_ant > 0 else None,
            "conserved_hits": int(con_hits), "conserved_expected": float(exp_con),
            "conserved_oe": float(con_hits / exp_con) if exp_con > 0 else None,
        }
        print(f"  plausibility: antigenic O/E={plausibility_results[acc]['antigenic_oe']}, "
              f"conserved O/E={plausibility_results[acc]['conserved_oe']}")

    # aggregate
    print("\n" + "="*72)
    print("STABILITY (median Spearman per virus)")
    print("="*72)
    stab_summary = {}
    for v in VIRUSES:
        vals = [r["spearman"] for r in stability_results.values()
                if r["virus"] == v and r["spearman"] is not None]
        if vals:
            stab_summary[v] = {"n": len(vals), "median": float(np.median(vals)),
                              "min": float(np.min(vals)), "max": float(np.max(vals))}
            print(f"  {v:15s} n={len(vals):2d}  median={np.median(vals):.3f}  "
                  f"range [{np.min(vals):.2f}, {np.max(vals):.2f}]")
    all_rho = [r["spearman"] for r in stability_results.values() if r["spearman"] is not None]
    print(f"  OVERALL median: {np.median(all_rho):.3f}")

    print("\n" + "="*72)
    print("FIDELITY (median margin drop per virus, model, deletion level)")
    print("="*72)
    for v in VIRUSES:
        recs = [r for r in fidelity_results.values() if r["virus"] == v]
        if not recs: continue
        print(f"\n  {v}:")
        for frac in DEL_LEVELS:
            c = np.median([r["deletion"][str(frac)]["cnn"] for r in recs])
            b = np.median([r["deletion"][str(frac)]["bilstm"] for r in recs])
            rc = np.median([r["deletion"][str(frac)]["random_cnn"] for r in recs])
            rb = np.median([r["deletion"][str(frac)]["random_bilstm"] for r in recs])
            print(f"    {frac*100:>4.0f}%  cnn={c:+.4f} (rand {rc:+.4f})  "
                  f"bilstm={b:+.4f} (rand {rb:+.4f})")

    print("\n" + "="*72)
    print("PLAUSIBILITY (per-virus O/E)")
    print("="*72)
    for v in VIRUSES:
        recs = [r for r in plausibility_results.values() if r["virus"] == v]
        if not recs: continue
        ant_oe = [r["antigenic_oe"] for r in recs if r["antigenic_oe"] is not None]
        con_oe = [r["conserved_oe"] for r in recs if r["conserved_oe"] is not None]
        print(f"\n  {v} (n={len(recs)}):")
        if ant_oe:
            print(f"    antigenic O/E:  median={np.median(ant_oe):.2f}  "
                  f"(n={len(ant_oe)}, values {[round(x,2) for x in ant_oe]})")
        if con_oe:
            print(f"    conserved O/E:  median={np.median(con_oe):.2f}  "
                  f"(n={len(con_oe)}, values {[round(x,2) for x in con_oe]})")

    out = {
        "stability": stability_results,
        "stability_summary": stab_summary,
        "fidelity": fidelity_results,
        "plausibility": plausibility_results,
    }
    json.dump(out, open(OUT / "evaluation.json", "w"), indent=2)
    print(f"\nwrote {OUT}/evaluation.json")


if __name__ == "__main__":
    main()
