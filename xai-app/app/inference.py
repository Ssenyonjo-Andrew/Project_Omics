import subprocess
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
import scipy.stats as stats

VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
VIR_TO_IDX = {v: i for i, v in enumerate(VIRUSES)}
N = len(VIRUSES)
WINDOW = 3000; STRIDE = 1500; BIN_BP = 100; DEV = "cpu"
MODELS_DIR = Path("../models")
GB_CACHE = Path("../data/raw/genbank"); GB_CACHE.mkdir(parents=True, exist_ok=True)

STRUCT_MAP = {
    "dengue":      {"pdb": "1OAN", "gene": "E"},
    "chikungunya": {"pdb": "3N42", "gene": "E1"},
    "norovirus":   {"pdb": "1IHM", "gene": "VP1"},
    "measles":     {"pdb": "2ZB6", "gene": "H"},
    "mpox":        {"pdb": "4K7Z", "gene": "DNA_POL"},
    "adeno_c":     {"pdb": "1X9P", "gene": "HEXON"}
}
REF_MAP = {
    "dengue": "NC_001477.1", "chikungunya": "NC_004162.2", "norovirus": "NC_029646.1",
    "measles": "NC_001498.1", "mpox": "NC_063383.1", "adeno_c": "NC_001405.1"
}

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

class GenomicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 128, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(128, N)
    def forward(self, x):
        e = self.emb(x); out = self.transformer(e)
        return self.fc(out.mean(dim=1))

@lru_cache(maxsize=1)
def _models():
    cnn = CNN(); cnn.load_state_dict(torch.load(MODELS_DIR / "cnn_v3.pt", map_location=DEV)); cnn.eval()
    bil = BiLSTM(); bil.load_state_dict(torch.load(MODELS_DIR / "bilstm_v3.pt", map_location=DEV)); bil.eval()
    trf = GenomicTransformer(); trf.load_state_dict(torch.load(MODELS_DIR / "transformer_v3.pt", map_location=DEV)); trf.eval()
    return cnn, bil, trf

def _windows(seq):
    return [seq[i:i+WINDOW] for i in range(0, len(seq)-WINDOW+1, STRIDE)] or [seq.ljust(WINDOW, 'N')]

def _onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in "ACGT": x["ACGT".index(c), i] = 1.0
    return x

def _kmers(s, k=6):
    ids = []
    for i in range(0, len(s)-k+1, k):
        idx, ok = 0, True
        for c in s[i:i+k]:
            if c not in "ACGT": ok = False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]

@torch.no_grad()
def predict_all(seq: str) -> dict:
    cnn, bil, trf = _models()
    ws = _windows(seq)
    X_cnn = torch.stack([torch.from_numpy(_onehot(w)) for w in ws])
    p_cnn = F.softmax(cnn(X_cnn), -1).mean(0).numpy()
    
    toks = [_kmers(w) for w in ws]; L = max(len(t) for t in toks)
    X_seq = torch.zeros(len(toks), L, dtype=torch.long)
    for i, t in enumerate(toks): X_seq[i, :len(t)] = torch.tensor(t)
    p_bil = F.softmax(bil(X_seq), -1).mean(0).numpy()
    p_trf = F.softmax(trf(X_seq), -1).mean(0).numpy()
    
    return {
        "cnn": {"virus": VIRUSES[int(np.argmax(p_cnn))], "probs": p_cnn.tolist()},
        "bilstm": {"virus": VIRUSES[int(np.argmax(p_bil))], "probs": p_bil.tolist()},
        "transformer": {"virus": VIRUSES[int(np.argmax(p_trf))], "probs": p_trf.tolist()},
    }

def attribute_all_methods(seq: str, model_name: str) -> dict:
    cnn, bil, trf = _models()
    model = cnn if model_name == "cnn" else (bil if model_name == "bilstm" else trf)
    base_pred = predict_all(seq)[model_name]
    cls = VIR_TO_IDX[base_pred["virus"]]
    
    n_bins = len(seq) // BIN_BP
    step = max(1, n_bins // 20)
    sample_bins = list(range(0, n_bins, step))[:20]
    ws = _windows(seq)
    n_win = len(ws)
    
    with torch.no_grad():
        if model_name == "cnn":
            X_base = torch.stack([torch.from_numpy(_onehot(w)) for w in ws])
            probs_base = F.softmax(model(X_base), -1).numpy()
        else:
            toks = [_kmers(w) for w in ws]; L = max(len(t) for t in toks)
            X_base = torch.zeros(len(toks), L, dtype=torch.long)
            for i, t in enumerate(toks): X_base[i, :len(t)] = torch.tensor(t)
            probs_base = F.softmax(model(X_base), -1).numpy()
        
    base_p = base_pred["probs"][cls]
    occl, perm, ig_a, shap = np.zeros(len(sample_bins)), np.zeros(len(sample_bins)), np.zeros(len(sample_bins)), np.zeros(len(sample_bins))
    rng = np.random.default_rng(42)
    shap_masks = rng.integers(0, 2, size=(30, len(sample_bins)))
    
    for idx, b in enumerate(sample_bins):
        bin_start, bin_end = b * BIN_BP, (b + 1) * BIN_BP
        overlap = [w for w in range(n_win) if bin_start < (w*STRIDE + WINDOW) and bin_end > (w*STRIDE)]
        if not overlap: continue
            
        # 1. Occlusion
        mut_ws = []
        for w in overlap:
            w_s = w * STRIDE
            seq_l = list(seq[w_s : w_s + WINDOW])
            for j in range(max(0, bin_start-w_s), min(WINDOW, bin_end-w_s)): seq_l[j] = "N"
            mut_ws.append("".join(seq_l))
        with torch.no_grad():
            if model_name == "cnn":
                X_m = torch.stack([torch.from_numpy(_onehot(w)) for w in mut_ws])
                p_m = F.softmax(model(X_m), -1).numpy()
            else:
                toks = [_kmers(w) for w in mut_ws]; L = max(len(t) for t in toks)
                X_m = torch.zeros(len(toks), L, dtype=torch.long)
                for i, t in enumerate(toks): X_m[i, :len(t)] = torch.tensor(t)
                p_m = F.softmax(model(X_m), -1).numpy()
            p_upd = np.copy(probs_base)
            for i, w in enumerate(overlap): p_upd[w] = p_m[i]
            occl[idx] = base_p - p_upd.mean(axis=0)[cls]

        # 2. Permutation
        mut_ws = []
        for w in overlap:
            w_s = w * STRIDE
            seq_l = list(seq[w_s : w_s + WINDOW])
            chunk = seq_l[max(0, bin_start-w_s) : min(WINDOW, bin_end-w_s)]
            rng.shuffle(chunk)
            seq_l[max(0, bin_start-w_s) : min(WINDOW, bin_end-w_s)] = chunk
            mut_ws.append("".join(seq_l))
        with torch.no_grad():
            if model_name == "cnn":
                X_m = torch.stack([torch.from_numpy(_onehot(w)) for w in mut_ws])
                p_m = F.softmax(model(X_m), -1).numpy()
            else:
                toks = [_kmers(w) for w in mut_ws]; L = max(len(t) for t in toks)
                X_m = torch.zeros(len(toks), L, dtype=torch.long)
                for i, t in enumerate(toks): X_m[i, :len(t)] = torch.tensor(t)
                p_m = F.softmax(model(X_m), -1).numpy()
            p_upd = np.copy(probs_base)
            for i, w in enumerate(overlap): p_upd[w] = p_m[i]
            perm[idx] = base_p - p_upd.mean(axis=0)[cls]

        # 3. Integrated Gradients
        w = overlap[0]
        win = seq[w*STRIDE : w*STRIDE + WINDOW].ljust(WINDOW, 'N')
        if model_name == "cnn":
            x = torch.from_numpy(_onehot(win)).unsqueeze(0).requires_grad_()
            logits = model(x); logits[0, cls].backward()
            grads = x.grad.data.numpy()[0]
            ig_map = (grads * x.detach().numpy()[0]).sum(0)
        else:
            tok = torch.tensor([_kmers(win)], dtype=torch.long)
            emb = model.emb(tok).detach().requires_grad_()
            if model_name == "bilstm":
                o, _ = model.lstm(emb); logits = model.fc(model.drop(o.mean(1)))
            else:
                o = model.transformer(emb); logits = model.fc(o.mean(1))
            logits[0, cls].backward()
            grads = emb.grad.data.numpy()[0]
            ig_map = np.repeat((grads * emb.detach().numpy()[0]).sum(-1), 6)[:WINDOW]
        l_start, l_end = max(0, bin_start - w*STRIDE), min(WINDOW, bin_end - w*STRIDE)
        ig_a[idx] = np.abs(ig_map[l_start:l_end]).mean()

    # 4. KernelSHAP
    for coal in shap_masks:
        mut_ws = []
        for w in range(n_win):
            w_s = w * STRIDE
            seq_l = list(seq[w_s : w_s + WINDOW])
            for i, b in enumerate(sample_bins):
                if coal[i] == 1:
                    for j in range(max(0, b*BIN_BP-w_s), min(WINDOW, (b+1)*BIN_BP-w_s)): seq_l[j] = "N"
            mut_ws.append("".join(seq_l))
        with torch.no_grad():
            if model_name == "cnn":
                X_m = torch.stack([torch.from_numpy(_onehot(w)) for w in mut_ws])
                p_m = F.softmax(model(X_m), -1).numpy()
            else:
                toks = [_kmers(w) for w in mut_ws]; L = max(len(t) for t in toks)
                X_m = torch.zeros(len(toks), L, dtype=torch.long)
                for i, t in enumerate(toks): X_m[i, :len(t)] = torch.tensor(t)
                p_m = F.softmax(model(X_m), -1).numpy()
        c_prob = p_m.mean(axis=0)[cls]
        for i, val in enumerate(coal):
            if val == 1: shap[i] += (base_p - c_prob)
    shap = shap / (shap_masks.sum(axis=0) + 1e-5)

    def norm(a):
        abs_a = np.abs(a)
        return (abs_a / abs_a.max()).tolist() if abs_a.max() > 0 else abs_a.tolist()

    return {
        "bins": sample_bins,
        "methods": {"occlusion": norm(occl), "permutation": norm(perm), "ig": norm(ig_a), "shap": norm(shap)},
        "importance": norm(occl)
    }

def fetch_genes(accession: str) -> list:
    cache = GB_CACHE / f"{accession}_ref.gb"
    if not cache.exists():
        r = subprocess.run(["efetch", "-db", "nuccore", "-id", accession, "-format", "gb"], capture_output=True, text=True, timeout=180)
        if r.returncode == 0: cache.write_text(r.stdout)
    genes = []
    if cache.exists():
        try:
            for rec in SeqIO.parse(str(cache), "genbank"):
                for feat in rec.features:
                    if feat.type in ("CDS", "mat_peptide"):
                        name = next((feat.qualifiers[k][0].upper() for k in ("gene", "product") if k in feat.qualifiers), "UNKNOWN")
                        genes.append({"name": name, "start": int(feat.location.start), "end": int(feat.location.end)})
        except: pass
    return genes

def map_bins_to_residues(bins, importance, genes, target):
    # Get top 5 indices sorted by importance
    top_indices = np.argsort(-np.array(importance))[:5]
    res = []
    g_coords = next((g for g in genes if target.upper() in g["name"].upper()), None)
    if not g_coords and genes: g_coords = {"start": min(g["start"] for g in genes), "end": max(g["end"] for g in genes)}
    if not g_coords: return []
    
    max_res = min(500, (g_coords["end"] - g_coords["start"]) // 3)
    for rank, idx in enumerate(top_indices):
        b = bins[idx]
        score = float(importance[idx])
        nt = b * BIN_BP
        aa = 1 if nt < g_coords["start"] else (max_res if nt > g_coords["end"] else ((nt - g_coords["start"]) // 3) + 1)
        resi = int((aa % max_res) + 1) if max_res > 0 else 1
        res.append({"resi": resi, "score": score, "rank": rank + 1})
    return res

def calculate_xai_metrics(seq: str, top_v: str) -> dict:
    attrs = {m: attribute_all_methods(seq, m) for m in ["cnn", "bilstm", "transformer"]}
    
    rho_cb = stats.spearmanr(attrs["cnn"]["importance"], attrs["bilstm"]["importance"]).correlation
    rho_ct = stats.spearmanr(attrs["cnn"]["importance"], attrs["transformer"]["importance"]).correlation
    rho_bt = stats.spearmanr(attrs["bilstm"]["importance"], attrs["transformer"]["importance"]).correlation
    rho = float(np.nanmean([rho_cb, rho_ct, rho_bt]))
    rho_matrix = [[1.0, rho_cb, rho_ct], [rho_cb, 1.0, rho_bt], [rho_ct, rho_bt, 1.0]]
    
    consensus = np.mean([attrs["cnn"]["importance"], attrs["bilstm"]["importance"], attrs["transformer"]["importance"]], axis=0)
    top_bins = np.argsort(-consensus)[:3]
    
    mask_seq = list(seq)
    for b_idx in top_bins:
        b = attrs["cnn"]["bins"][b_idx]
        for j in range(b*BIN_BP, min((b+1)*BIN_BP, len(seq))): mask_seq[j] = "N"
    masked_preds = predict_all("".join(mask_seq))
    
    v_idx = VIR_TO_IDX[top_v]
    orig_prob = predict_all(seq)["transformer"]["probs"][v_idx]
    mask_prob = masked_preds["transformer"]["probs"][v_idx]
    fidelity = orig_prob - mask_prob
    
    genes = fetch_genes(REF_MAP.get(top_v, ""))
    target_gene = STRUCT_MAP.get(top_v, {}).get("gene", "Unknown")
    
    hits = sum(1 for i in top_bins if any(attrs["cnn"]["bins"][i]*BIN_BP < g["end"] and (attrs["cnn"]["bins"][i]+1)*BIN_BP > g["start"] for g in genes))
    gene_len = sum(g["end"] - g["start"] for g in genes)
    oe = (hits / (3 * (gene_len / len(seq)))) if gene_len > 0 else 0.0
    
    res_idx_dict = {}
    for m in ["cnn", "bilstm", "transformer"]:
        try:
            res_idx_dict[m] = map_bins_to_residues(attrs[m]["bins"], attrs[m]["importance"], genes, target_gene)
        except:
            res_idx_dict[m] = []

    return {
        "stability": rho, "fidelity": float(fidelity), "plausibility": float(oe),
        "attributions": attrs, "evidence": {"orig_prob": float(orig_prob), "mask_prob": float(mask_prob), "target_gene": target_gene, "hits": hits},
        "genes": genes,
        "all_bins": {m: attrs[m]["bins"] for m in ["cnn", "bilstm", "transformer"]},
        "all_importance": {m: attrs[m]["importance"] for m in ["cnn", "bilstm", "transformer"]},
        "method_agreement": attrs["cnn"]["methods"],
        "rho_matrix": rho_matrix,
        "genome_bins": attrs["transformer"]["bins"],
        "genome_importance": attrs["transformer"]["importance"],
        "pdb_id": STRUCT_MAP.get(top_v, {}).get("pdb", "1OAN"),
        "all_residue_indices": res_idx_dict
    }
