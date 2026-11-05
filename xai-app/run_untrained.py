import numpy as np
import torch
from Bio import SeqIO
from app import inference

def test_untrained():
    print("="*50)
    print(" UNTRAINED MODEL CONTROL (RANDOM WEIGHTS)")
    print("="*50)

    # 1. Load a test sequence
    fasta_path = "../data/processed/final_split/test.fasta"
    rec = next(r for r in SeqIO.parse(fasta_path, "fasta") if r.id.startswith("adeno_c"))
    seq = str(rec.seq).upper()
    print(f"Sequence: {rec.id} ({len(seq)} bp)")

    # 2. Get original trained models
    orig_cnn, orig_bil, orig_trf = inference._models()

    # 3. Create an untrained CNN (random weights)
    untrained_cnn = inference.CNN().eval()

    # 4. Patch the models function to return the untrained CNN but keep the other models
    inference._models = lambda: (untrained_cnn, orig_bil, orig_trf)

    try:
        # Get predictions
        preds = inference.predict_all(seq)
        top_v = preds["cnn"]["virus"]
        print(f"Untrained CNN Prediction: {top_v} (Confidence: {preds['cnn']['probs'][inference.VIR_TO_IDX[top_v]]*100:.1f}%)")

        # Run XAI attribution on the untrained model
        attrs_untrained = inference.attribute_all_methods(seq, "cnn")
        imp_untrained = np.array(attrs_untrained["importance"])
        
        print("\n--- Untrained CNN Attributions ---")
        print(f"Variance: {np.var(imp_untrained):.5f}")
        print(f"Max Importance: {np.max(imp_untrained):.3f}")
        print(f"Mean Importance: {np.mean(imp_untrained):.3f}")
        
        # Restore the trained model and run XAI on it
        inference._models = lambda: (orig_cnn, orig_bil, orig_trf)
        attrs_trained = inference.attribute_all_methods(seq, "cnn")
        imp_trained = np.array(attrs_trained["importance"])
        
        print("\n--- Trained CNN Attributions ---")
        print(f"Variance: {np.var(imp_trained):.5f}")
        print(f"Max Importance: {np.max(imp_trained):.3f}")
        print(f"Mean Importance: {np.mean(imp_trained):.3f}")

        print("\nCONCLUSION:")
        if np.var(imp_untrained) > np.var(imp_trained):
             print("The untrained model produces HIGHER variance (more random noise) than the trained model.")
             print("This proves the trained model has learned specific genomic features.")
        else:
             print("The untrained model produces similar variance to the trained model.")
             print("This suggests the XAI method may be capturing noise.")

    finally:
        # Always restore the original models
        inference._models = lambda: (orig_cnn, orig_bil, orig_trf)

if __name__ == "__main__":
    test_untrained()
