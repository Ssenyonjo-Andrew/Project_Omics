"""Phase 2: balanced, stratified download."""
import json, random, subprocess, time, sys
from pathlib import Path
import yaml, pandas as pd

CFG = yaml.safe_load(open("config.yaml"))
CAP = CFG["project"]["cap_per_species"]
SEED = CFG["project"]["seed"]
LIMIT = CFG["sampling"]["summary_limit"]
BATCH = 100

META = {
    "dengue":       ("Dengue virus",       "Flaviviridae",    "Orthoflavivirus",   "Orthoflavivirus denguei"),
    "ebola":        ("Ebola virus",        "Filoviridae",     "Orthoebolavirus",   "Orthoebolavirus zairense"),
    "mers":         ("MERS-CoV",           "Coronaviridae",   "Betacoronavirus",   "Betacoronavirus cameli"),
    "yellow_fever": ("Yellow fever virus", "Flaviviridae",    "Orthoflavivirus",   "Orthoflavivirus flavi"),
    "sars_cov_2":   ("SARS-CoV-2",         "Coronaviridae",   "Betacoronavirus",   "Betacoronavirus pandemicum"),
    "zika":         ("Zika virus",         "Flaviviridae",    "Orthoflavivirus",   "Orthoflavivirus zikaense"),
    "marburg":      ("Marburg virus",      "Filoviridae",     "Orthomarburgvirus", "Orthomarburgvirus marburgense"),
    "lassa":        ("Lassa virus",        "Arenaviridae",    "Mammarenavirus",    "Mammarenavirus lassaense"),
    "nipah":        ("Nipah virus",        "Paramyxoviridae", "Henipavirus",       "Henipavirus nipahense"),
}

def load_taxa():
    taxa = {}
    for line in open("data/raw/resolved_taxa.tsv"):
        slug, taxon, _ = [x.strip() for x in line.rstrip("\n").split("\t")]
        if taxon != "NONE":
            taxa[slug] = taxon
    return taxa

def fetch_summary(slug, taxon):
    p = Path(f"data/raw/expanded/{slug}.jsonl")
    if p.exists() and p.stat().st_size > 0:
        return p
    with open(p, "w") as f:
        subprocess.run(["datasets","summary","virus","genome","taxon",taxon,
                        "--complete-only","--limit",str(LIMIT),"--as-json-lines"],
                       stdout=f, stderr=subprocess.DEVNULL, check=True)
    return p

def bucket(rec, slug):
    year = (rec.get("isolate",{}).get("collection_date","") or "unknown")[:4]
    key = year or "unknown"
    if slug == "dengue":
        s = rec.get("isolate",{}).get("name","")
        for st in ("1","2","3","4"):
            if f"type {st}" in s.lower() or f"denv-{st}" in s.lower():
                key += f"|serotype{st}"; break
    return key

def stratified(records, slug, cap):
    rng = random.Random(SEED)
    buckets = {}
    for r in records:
        buckets.setdefault(bucket(r, slug), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    out = []
    while len(out) < cap and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < cap:
                out.append(buckets[k].pop())
    return out

def efetch(slug, accs):
    fa = Path(f"data/raw/expanded/{slug}.fasta")
    if fa.exists() and fa.stat().st_size > 0:
        n = sum(1 for l in open(fa) if l.startswith(">"))
        if n >= len(accs):
            print(f"  {slug:14s} fasta already has {n} sequences, skipping")
            return fa
    with open(fa, "w") as f:
        for i in range(0, len(accs), BATCH):
            r = subprocess.run(["efetch","-db","nucleotide","-id",",".join(accs[i:i+BATCH]),
                                "-format","fasta"], capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                f.write(r.stdout)
            time.sleep(0.4)
    return fa

def main():
    taxa = load_taxa()
    rows = []
    for slug, taxon in taxa.items():
        vname, family, genus, species = META[slug]
        jl = fetch_summary(slug, taxon)
        recs = [json.loads(l) for l in open(jl) if l.strip()]
        sampled = stratified(recs, slug, CAP)
        if len(sampled) < CAP:
            sys.exit(f"FAIL: {slug} returned {len(sampled)} < CAP={CAP}")
        accs = [r["accession"] for r in sampled]
        fa = efetch(slug, accs)
        print(f"  {slug:14s} {len(accs):3d}  -> {fa}")
        for a in accs:
            rows.append(dict(accession=a, sequence_file_path=str(fa), virus_name=vname,
                             family=family, genus=genus, species=species,
                             cluster_id=pd.NA, split=pd.NA))
    df = pd.DataFrame(rows)
    df = df[["accession","sequence_file_path","virus_name","family","genus","species","cluster_id","split"]]
    df.to_csv("data/processed/master_expanded.csv", index=False)
    counts = df["virus_name"].value_counts()
    assert (counts == CAP).all(), f"IMBALANCE: {counts.to_dict()}"
    print(f"\nWrote {len(df)} rows. Every species = {CAP}. PASS")

if __name__ == "__main__":
    main()
