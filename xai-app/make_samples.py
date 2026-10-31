import os
from Bio import SeqIO

# Path to the test set (we go up one folder to find data/)
test_fasta = "../data/processed/final_split/test.fasta"
out_dir = "test_samples"
os.makedirs(out_dir, exist_ok=True)

found = set()
for rec in SeqIO.parse(test_fasta, "fasta"):
    virus = rec.id.split("|")[0]
    if virus not in found:
        found.add(virus)
        out_path = os.path.join(out_dir, f"{virus}_sample.fasta")
        with open(out_path, "w") as f:
            f.write(f">{rec.id}\n{rec.seq}\n")
        print(f"✅ Saved: {out_path}")
        
    if len(found) == 6:
        break
