import json
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from app import inference
import scipy.stats as stats

VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
N_PER_VIRUS = 20
OUT = Path("results/benchmark"); OUT.mkdir(parents=True, exist_ok=True)

def get_baselines(seq_len=35000, num_bins=350):
    # 1. Stability Baseline: Correlation of two random arrays
    random_stab_scores = []
    for _ in range(1000):
        rand1 = np.random.rand(num_bins)
        rand2 = np.random.rand(num_bins)
        rho = np.corrcoef(rand1, rand2)[0, 1]
        random_stab_scores.append(rho)
    
    return {
        "stability_mean": float(np.mean(random_stab_scores)),
        "stability_95th_percentile": float(np.percentile(random_stab_scores, 95)),
        "fidelity_random_drop": 0.0,
        "plausibility_expected_oe": 1.0
    }

def main():
    print("Starting Benchmark...")
    results = []
    baselines = get_baselines()
    json.dump(baselines, open(OUT / "baselines.json", "w"), indent=2)
    print(f"Baselines: Stability 95th Percentile = {baselines['stability_95th_percentile']:.3f}")
    
    fasta_path = Path("../data/processed/final_split/test.fasta")
    if not fasta_path.exists():
        print(f"ERROR: Could not find {fasta_path}. Check your paths.")
        return

    for virus in VIRUSES:
        # Get sequences for this virus
        records = [r for r in SeqIO.parse(fasta_path, "fasta") if r.id.startswith(virus)][:N_PER_VIRUS]
        print(f"\nProcessing {virus} ({len(records)} sequences)...")
        
        for i, rec in enumerate(records):
            seq = str(rec.seq).upper()
            try:
                print(f"  [{i+1}/{len(records)}] {rec.id}...", end="", flush=True)
                # 1. Run the XAI pipeline
                preds = inference.predict_all(seq)
                top_v = preds["transformer"]["virus"]
                metrics = inference.calculate_xai_metrics(seq, top_v)
                
                # 2. Save results
                results.append({
                    "virus": virus,
                    "accession": rec.id,
                    "length": len(seq),
                    "stability": metrics["stability"],
                    "fidelity": metrics["fidelity"],
                    "plausibility": metrics["plausibility"]
                })
                print(" Done.")
            except Exception as e:
                print(f" Error: {e}")
                
    # 3. Calculate averages
    df = pd.DataFrame(results)
    df.to_csv(OUT / "benchmark_raw.csv", index=False)
    
    summary = df.groupby("virus").agg({
        "stability": ["mean", "std"],
        "fidelity": ["mean", "std"],
        "plausibility": ["mean", "std"]
    }).round(4)
    
    summary.to_csv(OUT / "benchmark_summary.csv")
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS (Mean ± SD)")
    print("="*50)
    print(summary.to_string())
    print(f"\nSaved all results to {OUT}/")

if __name__ == "__main__":
    main()
