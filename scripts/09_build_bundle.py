"""Precompute prototype data bundle from existing results."""
import json
import numpy as np
from pathlib import Path
from Bio import SeqIO

ATTR = Path("results/attr_v2")
EVAL = Path("results/eval_v2")
OUT = Path("results/prototype"); OUT.mkdir(exist_ok=True)

predictions = json.load(open(ATTR / "predictions.json"))
evaluation = json.load(open(EVAL / "evaluation.json"))

viruses = ["dengue", "chikungunya", "hcv", "norovirus"]

seqs = {}
for v in viruses:
    fasta = Path(f"data/processed/split/{v}_test.fasta")
    for rec in SeqIO.parse(fasta, "fasta"):
        seqs[rec.id] = str(rec.seq).upper()

bundle = {
    "viruses": viruses,
    "sequences": {},
    "stability_summary": evaluation.get("stability_summary", {}),
    "per_virus_stability": {},
    "per_virus_fidelity": {},
    "per_virus_plausibility": {},
    "training_results": json.load(open("results/train_results.json")),
}

for v in viruses:
    stab_vals = [r["spearman"] for r in evaluation["stability"].values()
                 if r["virus"] == v and r["spearman"] is not None]
    if stab_vals:
        bundle["per_virus_stability"][v] = {
            "n": len(stab_vals),
            "median": float(np.median(stab_vals)),
            "min": float(np.min(stab_vals)),
            "max": float(np.max(stab_vals)),
        }

    fid_recs = [r for r in evaluation["fidelity"].values() if r["virus"] == v]
    if fid_recs:
        bundle["per_virus_fidelity"][v] = {}
        for frac in ["0.01", "0.05", "0.1", "0.2"]:
            bundle["per_virus_fidelity"][v][frac] = {
                "cnn": float(np.median([r["deletion"][frac]["cnn"] for r in fid_recs])),
                "bilstm": float(np.median([r["deletion"][frac]["bilstm"] for r in fid_recs])),
                "random_cnn": float(np.median([r["deletion"][frac]["random_cnn"] for r in fid_recs])),
                "random_bilstm": float(np.median([r["deletion"][frac]["random_bilstm"] for r in fid_recs])),
            }

    pl_recs = [r for r in evaluation["plausibility"].values() if r["virus"] == v]
    if pl_recs:
        ant = [r["antigenic_oe"] for r in pl_recs if r["antigenic_oe"] is not None]
        con = [r["conserved_oe"] for r in pl_recs if r["conserved_oe"] is not None]
        bundle["per_virus_plausibility"][v] = {
            "n_annotated": len(ant),
            "antigenic_oe_values": ant,
            "conserved_oe_values": con,
        }

for acc, pred in predictions.items():
    seq = seqs.get(acc)
    if not seq:
        continue

    cons_cnn = [ATTR / "cnn" / m / f"{acc}.npy" for m in
                ["occlusion", "permutation", "ig", "kernelshap"]]
    cons_bil = [ATTR / "bilstm" / m / f"{acc}.npy" for m in
                ["occlusion", "permutation", "ig", "kernelshap"]]

    def load_mean(paths):
        arrs = [np.abs(np.load(p)) for p in paths if p.exists()]
        if not arrs:
            return None
        return np.mean(np.stack(arrs), axis=0).tolist()

    cons_cnn_arr = load_mean(cons_cnn)
    cons_bil_arr = load_mean(cons_bil)

    cons_combined = None
    if cons_cnn_arr and cons_bil_arr:
        a = np.array(cons_cnn_arr); b = np.array(cons_bil_arr)
        cons_combined = ((a / (a.max() + 1e-9)) + (b / (b.max() + 1e-9))).tolist()

    bundle["sequences"][acc] = {
        "virus": pred["true_virus"],
        "length": pred["sequence_length"],
        "cnn_pred": pred["cnn_pred"],
        "cnn_probs": pred["cnn_probs"],
        "bilstm_pred": pred["bilstm_pred"],
        "bilstm_probs": pred["bilstm_probs"],
        "consensus_cnn": cons_cnn_arr,
        "consensus_bilstm": cons_bil_arr,
        "consensus_combined": cons_combined,
        "stability": evaluation["stability"].get(acc, {}).get("spearman"),
        "fidelity": evaluation["fidelity"].get(acc, {}).get("deletion", {}),
        "plausibility": evaluation["plausibility"].get(acc, {}),
    }

json.dump(bundle, open(OUT / "bundle.json", "w"))
print(f"wrote {OUT / 'bundle.json'}")
print(f"  {len(bundle['sequences'])} sequences")
print(f"  {len(bundle['per_virus_stability'])} stability summaries")
print(f"  {len(bundle['per_virus_fidelity'])} fidelity summaries")
