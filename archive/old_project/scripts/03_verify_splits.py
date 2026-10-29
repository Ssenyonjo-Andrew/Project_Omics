import subprocess
import pandas as pd
from pathlib import Path

MASTER = "data/processed/master.csv"

def write_split_fasta(df, path):
    keep = set(df["accession"].str.split(".").str[0])
    out, write = [], False
    for line in open("data/processed/sequences.fasta"):
        if line.startswith(">"):
            acc = line[1:].split(".")[0]
            write = acc in keep
        if write:
            out.append(line)
    Path(path).write_text("".join(out))

def main():
    df = pd.read_csv(MASTER)
    write_split_fasta(df[df.split=="train"], "data/processed/train.fasta")
    write_split_fasta(df[df.split=="test"],  "data/processed/test.fasta")

    print("Running cd-hit-2d (test vs train, 90% identity)...")
    result = subprocess.run(
        ["cd-hit-2d",
         "-i", "data/processed/test.fasta",
         "-i2", "data/processed/train.fasta",
         "-o", "data/processed/test_vs_train_90.fasta",
         "-c", "0.90", "-n", "8", "-M", "0", "-T", "0"],
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    # cd-hit-2d prints "N finished  M clusters" where N is matching pairs
    matches = 0
    for line in output.splitlines():
        if "finished" in line and "clusters" in line:
            matches = int(line.split()[0])

    print(f"\nTest-vs-train matches at 90%: {matches}")
    if matches == 0:
        print("PASS - no leakage")
    else:
        print("FAIL - leakage detected; regenerate splits")
        raise SystemExit(1)

if __name__ == "__main__":
    main()