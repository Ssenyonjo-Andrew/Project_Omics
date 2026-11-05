import random
import numpy as np
from Bio import SeqIO
from app import inference
import scipy.stats as stats

def test_baselines(virus_name, accession_prefix):
    print(f"\n" + "="*50)
    print(f" RUNNING BASELINES FOR: {virus_name.upper()}")
    print("="*50)

    # 1. Load a test sequence
    fasta_path = "../data/processed/final_split/test.fasta"
    records = [r for r in SeqIO.parse(fasta_path, "fasta") if r.id.startswith(accession_prefix)]
    if not records:
        print("Could not find sequence.")
        return
    rec = records[0]
    seq = str(rec.seq).upper()
    seq_len = len(seq)
    
    # 2. Get actual model predictions and metrics
    preds = inference.predict_all(seq)
    top_v = preds["transformer"]["virus"]
    actual_metrics = inference.calculate_xai_metrics(seq, top_v)
    
    actual_fid = actual_metrics["fidelity"] * 100
    actual_plaus = actual_metrics["plausibility"]
    
    print(f"Sequence: {rec.id} ({seq_len} bp)")
    print(f"Actual Fidelity Drop: {actual_fid:.2f}%")
    print(f"Actual Plausibility (O/E): {actual_plaus:.3f}")

    # --- FIDELITY BASELINE ---
    print("\n[Calculating Fidelity Random Baseline...]")
    # The actual test masks the top 3 consensus bins (3 * 100bp = 300bp total)
    n_mask_bins = 3 
    base_prob = preds["transformer"]["probs"][inference.VIR_TO_IDX[top_v]]
    random_drops = []
    
    for _ in range(100):
        mask_seq = list(seq)
        # Pick 3 random 100-bp bins
        random_bins = random.sample(range(seq_len // 100), n_mask_bins)
        for b in random_bins:
            for j in range(b*100, min((b+1)*100, seq_len)):
                mask_seq[j] = "N"
                
        masked_preds = inference.predict_all("".join(mask_seq))
        masked_prob = masked_preds["transformer"]["probs"][inference.VIR_TO_IDX[top_v]]
        drop = (base_prob - masked_prob) * 100
        random_drops.append(drop)
        
    rand_fid_mean = np.mean(random_drops)
    rand_fid_95th = np.percentile(random_drops, 95)
    
    print(f"Random Baseline Fidelity Drop: Mean={rand_fid_mean:.2f}%, 95th Percentile={rand_fid_95th:.2f}%")
    if actual_fid <= rand_fid_95th:
        print("RESULT: The actual fidelity drop is NOT significantly different from random masking (p > 0.05).")
    else:
        print("RESULT: The actual fidelity drop IS significantly higher than random masking.")

    # --- PLAUSIBILITY BASELINE ---
    print("\n[Calculating Plausibility Permutation Baseline...]")
    genes = inference.fetch_genes(inference.REF_MAP.get(top_v, ""))
    gene_len = sum(g["end"] - g["start"] for g in genes)
    
    if gene_len > 0:
        # Actual model hit 3 bins inside genes (or whatever hits it got)
        # We will randomly place 3 bins 1000 times
        random_oes = []
        for _ in range(1000):
            hits = 0
            for _ in range(3):
                b = random.randint(0, (seq_len // 100) - 1)
                bin_start, bin_end = b * 100, (b + 1) * 100
                if any(bin_start < g["end"] and bin_end > g["start"] for g in genes):
                    hits += 1
            oe = hits / (3 * (gene_len / seq_len))
            random_oes.append(oe)
            
        rand_plaus_mean = np.mean(random_oes)
        rand_plaus_95th = np.percentile(random_oes, 95)
        p_value = sum(1 for x in random_oes if x >= actual_plaus) / 1000
        
        print(f"Random Baseline Plausibility (O/E): Mean={rand_plaus_mean:.2f}, 95th Percentile={rand_plaus_95th:.2f}")
        print(f"Permutation p-value: {p_value:.3f}")
        if p_value > 0.05:
            print("RESULT: The actual plausibility is NOT significantly different from random placement (p > 0.05).")
        else:
            print("RESULT: The actual plausibility IS significantly different from random placement.")
    else:
        print("No genes found for this virus to calculate plausibility.")

if __name__ == "__main__":
    # Test Adenovirus (which had high stability but low plausibility)
    test_baselines("Adenovirus C", "adeno_c")
    # Test Norovirus (which had the worst plausibility score of 0.008)
    test_baselines("Norovirus", "norovirus")
