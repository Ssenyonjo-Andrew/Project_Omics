"""Phase 3b: train DNABERT-2 linear probe only. Reuses saved CNN/BiLSTM results."""
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
from transformers import AutoTokenizer, AutoModel

# Bypass triton/flash_attn import check: we use eager attention,
# so neither is ever called at runtime.
import transformers.dynamic_module_utils as _dmu
_orig_check = _dmu.check_imports
def _patched(fn):
    try:
        return _orig_check(fn)
    except ImportError as e:
        msg = str(e).lower()
        # Only swallow if the missing packages are triton/flash_attn
        if ("triton" in msg or "flash_attn" in msg or "flash-attn" in msg) \
           and "einops" not in msg:
            return []
        raise
_dmu.check_imports = _patched

CFG = yaml.safe_load(open("config.yaml"))
SEED = CFG["project"]["seed"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

FAMILIES = CFG["models"]["families"]
FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N_CLASSES = len(FAMILIES)
CKPT_DIR = Path(CFG["training"]["checkpoint_dir"])
RES_DIR  = Path(CFG["training"]["results_dir"])
DEV = "cpu"
print(f"device: {DEV}")


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


def main():
    splits = load_split()
    print(f"train: {len(splits['train'])} sequences, val: {len(splits['val'])} sequences")

    d = CFG["models"]["dnabert2"]
    from transformers import AutoConfig
    tok = AutoTokenizer.from_pretrained(d["name"], trust_remote_code=True)
    config = AutoConfig.from_pretrained(d["name"], trust_remote_code=True)
    # DNABERT-2 custom code reads config.pad_token_id; newer transformers
    # does not populate it by default.
    config.pad_token_id = 0
    if getattr(config, "eos_token_id", None) is None:
        config.eos_token_id = 1
    if getattr(config, "bos_token_id", None) is None:
        config.bos_token_id = 2
    bert = AutoModel.from_pretrained(
        d["name"], config=config, trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False
    ).to(DEV)
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

    print("extracting train embeddings...")
    Xtr, ytr, wtr, _ = embed(splits["train"]); print(f"  {len(Xtr)} windows")
    print("extracting val embeddings...")
    Xva, yva, wva, sva = embed(splits["val"]); print(f"  {len(Xva)} windows")

    head = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, N_CLASSES),
    ).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best_f1, best_state = -1.0, None
    for ep in range(1, 31):
        head.train(); perm = np.random.permutation(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]
            xb = torch.from_numpy(Xtr[idx]).to(DEV)
            yb = torch.from_numpy(ytr[idx]).to(DEV)
            wb = torch.from_numpy(wtr[idx]).to(DEV)
            ce = F.cross_entropy(head(xb), yb, reduction="none", label_smoothing=0.1)
            loss = (ce * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(Xva).to(DEV)).cpu().numpy()
        by_seq = {}
        for i, sidx in enumerate(sva.tolist()):
            by_seq.setdefault(sidx, []).append(logits[i])
        yt, yp = [], []
        for sidx, (_, _, fam) in enumerate(splits["val"]):
            if sidx not in by_seq: continue
            yt.append(FAM2IDX[fam])
            yp.append(int(np.argmax(np.mean(np.stack(by_seq[sidx]), 0))))
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        if ep % 5 == 0 or ep == 30:
            print(f"  ep {ep:2d}  val_macro_f1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    torch.save({"head": head.state_dict(), "bert_name": d["name"]},
               CKPT_DIR / "dnabert2_probe.pt")
    print(f"  saved {CKPT_DIR}/dnabert2_probe.pt (best val macro-F1 = {best_f1:.4f})")

    mf = RES_DIR / "train_metrics.json"
    metrics = json.load(open(mf)) if mf.exists() else {}
    metrics["dnabert2"] = {
        "val_macro_f1": float(best_f1),
        "mode": "linear_probe_frozen_bert",
        "passes_gate": float(best_f1) > 0.20,
    }
    json.dump(metrics, open(mf, "w"), indent=2)
    print(f"\nupdated {mf}")
    for name, r in metrics.items():
        print(f"  {name:10s}  val_macro_f1 = {r.get('val_macro_f1', 'n/a')}  gate = {r.get('passes_gate', 'n/a')}")


if __name__ == "__main__":
    main()
