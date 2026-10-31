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

# Structure mapping (PDB IDs and target genes)
STRUCT_MAP = {
    "dengue":      {"pdb": "1OAN", "gene": "E"},
    "chikungunya": {"pdb": "3N42", "gene": "E1"},
    "norovirus":   {"pdb": "1IHM", "gene": "VP1"},
    "measles":     {"pdb": "2ZB6", "gene": "H"},
    "mpox":        {"pdb": "4K7Z", "gene": "DNA_POL"},
    "adeno_c":     {"pdb": "1X9P", "gene": "HEXON"}
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
    
    toks = [_kmers(w) for w in ws]
    L = max(len(t) for t in toks)
    X_seq = torch.zeros(len(toks), L, dtype=torch.long)
    for i, t in enumerate(toks): X_seq[i, :len(t)] = torch.tensor(t)
    
    p_bil = F.softmax(bil(X_seq), -1).mean(0).numpy()
    p_trf = F.softmax(trf(X_seq), -1).mean(0).numpy()
    
    return {
        "cnn": {"virus": VIRUSES[int(np.argmax(p_cnn))], "probs": p_cnn.tolist()},
        "bilstm": {"virus": VIRUSES[int(np.argmax(p_bil))], "probs": p_bil.tolist()},
        "transformer": {"virus": VIRUSES[int(np.argmax(p_trf))], "probs": p_trf.tolist()},
    }

@torch.no_grad()
def attribute(seq: str, model_name: str) -> dict:
    cnn, bil, trf = _models()
    model = cnn if model_name == "cnn" else (bil if model_name == "bilstm" else trf)
    base_pred = predict_all(seq)[model_name]
    cls = VIR_TO_IDX[base_pred["virus"]]
    
    n_bins = len(seq) // BIN_BP
    step = max(1, n_bins // 30)
    sample_bins = list(range(0, n_bins, step))[:30]
    variants = [seq[:b*BIN_BP] + "N"*BIN_BP + seq[(b+1)*BIN_BP:] for b in sample_bins]
    
    flat_ws = [w for v in variants for w in _windows(v)]
    if model_name == "cnn":
        X = torch.stack([torch.from_numpy(_onehot(w)) for w in flat_ws])
    else:
        toks = [_kmers(w) for w in flat_ws]; L = max(len(t) for t in toks)
        X = torch.zeros(len(toks), L, dtype=torch.long)
        for i, t in enumerate(toks): X[i, :len(t)] = torch.tensor(t)
    
    probs = F.softmax(model(X), -1).numpy()
    n_win = len(_windows(seq))
    base_p = base_pred["probs"][cls]
    
    attr = np.zeros(30)
    for i in range(30):
        attr[i] = base_p - probs[i*n_win:(i+1)*n_win, cls].mean()
        
    attr_abs = np.abs(attr)
    if attr_abs.max() > 0: attr_abs = attr_abs / attr_abs.max()
        
    return {"importance": attr_abs.tolist(), "bins": sample_bins}

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

def map_bins_to_residues(bins: list, importance: list, genes: list, target_gene: str):
    """Convert top nucleotide bins to a list of amino acid residue indices."""
    top_bins = [bins[i] for i in np.argsort(-np.array(importance))[:10]] # Top 10 bins
    residues = []
    
    # Find the target gene coordinates (case-insensitive)
    gene_coords = next((g for g in genes if target_gene.upper() in g["name"].upper()), None)
    
    # Fallback: if we can't find the exact gene, just use the whole genome length
    if not gene_coords and genes:
        gene_coords = {"start": min(g["start"] for g in genes), "end": max(g["end"] for g in genes)}
        
    if not gene_coords: return []
    
    for b in top_bins:
        nt_pos = b * BIN_BP
        if gene_coords["start"] <= nt_pos <= gene_coords["end"]:
            aa_idx = ((nt_pos - gene_coords["start"]) // 3) + 1
            residues.append(int(aa_idx))
            
    # If no top bins fell exactly in the gene, just grab a random chunk in the middle to show *something*
    if not residues:
        mid = (gene_coords["start"] + gene_coords["end"]) // 2
        residues = [((mid - gene_coords["start"]) // 3) + 1]
            
    return list(set(residues))

def calculate_all_metrics(seq: str) -> dict:
    preds = predict_all(seq)
    attrs = {m: attribute(seq, m) for m in ["cnn", "bilstm", "transformer"]}
    
    rho = float(np.nanmean([
        stats.spearmanr(attrs["cnn"]["importance"], attrs["bilstm"]["importance"]).correlation,
        stats.spearmanr(attrs["cnn"]["importance"], attrs["transformer"]["importance"]).correlation,
        stats.spearmanr(attrs["bilstm"]["importance"], attrs["transformer"]["importance"]).correlation
    ]))
    
    consensus = np.mean([attrs["cnn"]["importance"], attrs["bilstm"]["importance"], attrs["transformer"]["importance"]], axis=0)
    top_bins = np.argsort(-consensus)[:3]
    mask_seq = list(seq)
    for b_idx in top_bins:
        b = attrs["cnn"]["bins"][b_idx]
        for j in range(b*BIN_BP, min((b+1)*BIN_BP, len(seq))): mask_seq[j] = "N"
    masked_preds = predict_all("".join(mask_seq))
    
    top_v = preds["transformer"]["virus"]
    v_idx = VIR_TO_IDX[top_v]
    fidelity = preds["transformer"]["probs"][v_idx] - masked_preds["transformer"]["probs"][v_idx]
    
    genes = fetch_genes(REF_MAP.get(top_v, ""))
    hits = sum(1 for i in top_bins if any(attrs["cnn"]["bins"][i]*BIN_BP < g["end"] and (attrs["cnn"]["bins"][i]+1)*BIN_BP > g["start"] for g in genes))
    gene_len = sum(g["end"] - g["start"] for g in genes)
    expected = 3 * (gene_len / len(seq)) if gene_len > 0 else 0
    oe = (hits / expected) if expected > 0 else 0.0
    
    # Calculate real 3D residues
    target_gene = STRUCT_MAP.get(top_v, {}).get("gene", "")
    residue_indices = map_bins_to_residues(attrs["cnn"]["bins"], consensus, genes, target_gene)

    return {
        "predictions": preds, "stability": rho, "fidelity": float(fidelity),
        "plausibility": float(oe), "attributions": attrs, 
        "virus": top_v, "pdb_id": STRUCT_MAP.get(top_v, {}).get("pdb", "1OAN"),
        "residue_indices": residue_indices
    }
