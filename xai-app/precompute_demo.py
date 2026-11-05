import json
from pathlib import Path
from Bio import SeqIO
from app import inference

OUT = Path("app/static/demo_cache"); OUT.mkdir(parents=True, exist_ok=True)
VIRUSES = ["adeno_c", "chikungunya", "dengue", "measles", "mpox", "norovirus"]
FASTA = "../data/processed/final_split/test.fasta"

print("Generating precomputed demo data...")
records = list(SeqIO.parse(FASTA, "fasta"))

for v in VIRUSES:
    # Find one sequence for this virus
    rec = next((r for r in records if r.id.startswith(v)), None)
    if not rec:
        print(f"⚠️ No sequence found for {v}")
        continue
        
    seq = str(rec.seq).upper()
    print(f"Processing {v} ({rec.id})...")
    
    # Run the pipeline
    preds = inference.predict_all(seq)
    top_v = preds["transformer"]["virus"]
    metrics = inference.calculate_xai_metrics(seq, top_v)
    
    # Construct the exact response the frontend expects
    response = {
        "status": "done",
        "step": "done",
        "predictions": preds,
        "metrics": metrics,
        "error": None
    }
    
    # Save to JSON
    out_file = OUT / f"{v}_cache.json"
    json.dump(response, open(out_file, "w"))
    print(f"  Saved to {out_file}")

print("✅ All demo files generated!")
