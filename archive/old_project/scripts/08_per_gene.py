"""Per-gene plausibility: aggregate top-attributed bins by gene across all
sequences in each family, and test observed-vs-expected enrichment."""
import json, subprocess, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, yaml
from scipy.stats import hypergeom
from Bio import SeqIO

CFG = yaml.safe_load(open("config.yaml"))
RES = Path(CFG["training"]["results_dir"])
ATTR = RES / "attr"; CONS = RES / "consensus"
OUT = RES / "eval"; OUT.mkdir(parents=True, exist_ok=True)
BIN_BP = 100
TOP_FRAC = 0.10   # top 10% of bins per sequence


def slug_to_family():
    df = pd.read_csv("data/processed/master_final.csv")
    df = df[df["split"] == "test"]
    return {r["accession"]: r["family"] for _, r in df.iterrows()}


def fetch_per_gene(accession):
    """Return {gene_name: [(start, end), ...]} using CDS and mat_peptide.
    Caches GenBank to data/raw/genbank/<acc>.gb. A zero-byte cache file
    means the record is known-unavailable and is skipped silently."""
    cache = Path(f"data/raw/genbank/{accession}.gb")
    path = None
    cleanup = False

    if cache.exists():
        if cache.stat().st_size == 0:
            return {}
        path = str(cache)
    else:
        import tempfile
        tf = tempfile.NamedTemporaryFile(mode="w+", suffix=".gb", delete=False)
        path = tf.name; tf.close(); cleanup = True
        try:
            r = subprocess.run(
                ["efetch","-db","nuccore","-id",accession,"-format","gb"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            Path(path).unlink(missing_ok=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.touch()
            return {}
        if r.returncode != 0 or not r.stdout.strip():
            Path(path).unlink(missing_ok=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.touch()
            return {}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(r.stdout)
        path = str(cache)
        cleanup = False

    genes = defaultdict(list)
    try:
        for rec in SeqIO.parse(path, "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS", "mat_peptide"):
                    continue
                name = None
                for key in ("gene", "product", "standard_name"):
                    if key in feat.qualifiers:
                        name = feat.qualifiers[key][0]; break
                if not name: continue
                name = name.upper().strip()
                for suffix in (" PROTEIN", " POLYPROTEIN", " (PP)"):
                    if name.endswith(suffix):
                        name = name[:-len(suffix)].strip()
                try:
                    genes[name].append((int(feat.location.start),
                                        int(feat.location.end)))
                except Exception:
                    for part in feat.location.parts:
                        genes[name].append((int(part.start), int(part.end)))
    finally:
        if cleanup:
            Path(path).unlink(missing_ok=True)
    return dict(genes)


def bin_coverage(spans, seq_len, n_bins):
    """Boolean array: is each bin covered by any span in `spans`?"""
    arr = np.zeros(n_bins, dtype=bool)
    for s, e in spans:
        sb = max(0, s // BIN_BP)
        eb = min(n_bins, (e - 1) // BIN_BP + 1)
        arr[sb:eb] = True
    return arr


def main():
    fam_of = slug_to_family()
    meta = json.load(open(RES / "attr_metadata.json"))
    accs = meta["sample_accessions"]

    # aggregates: family -> gene -> counters
    agg = defaultdict(lambda: defaultdict(lambda: {
        "gene_bins_total": 0, "seq_bins_total": 0,
        "top_bins_in_gene": 0, "top_bins_total": 0,
        "n_sequences_with_gene": 0,
    }))

    for i, acc in enumerate(accs):
        cons_p = CONS / f"{acc}.npy"
        if not cons_p.exists(): continue
        cons = np.load(cons_p); n_bins = len(cons)
        seq_len = n_bins * BIN_BP
        fam = fam_of.get(acc)
        if fam is None: continue
        print(f"[{i+1}/{len(accs)}] {acc} {fam}")
        genes = fetch_per_gene(acc)
        if not genes:
            print(f"  no gene annotations"); continue

        # decide which feature class to use:
        # if mat_peptide-derived names exist (short, like 'E','NS5'), use those
        # otherwise fall back to CDS gene names (NP, GP, S, M, N, etc.)
        k = max(1, int(round(TOP_FRAC * n_bins)))
        top = np.argsort(cons)[:k]            # lowest consensus-rank = most important
        top_bin = np.zeros(n_bins, dtype=bool); top_bin[top] = True

        # filter out "POLY"/"ORF1AB" as whole-genome entries only when a
        # finer-grained mat_peptide is available
        peptide_names = {"C","PRM","PR","M","E","NS1","NS2A","NS2B",
                          "NS3","NS4A","NS4B","NS4B","NS5","2K"}
        has_peptides = any(n in peptide_names for n in genes)
        chosen = {n: s for n, s in genes.items()
                  if not (has_peptides and n in
                          ("POLY","POLYPROTEIN","ORF1AB","ORF1A"))}

        for gene, spans in chosen.items():
            cov = bin_coverage(spans, seq_len, n_bins)
            a = agg[fam][gene]
            a["gene_bins_total"] += int(cov.sum())
            a["seq_bins_total"] += n_bins
            a["top_bins_in_gene"] += int((cov & top_bin).sum())
            a["top_bins_total"] += k
            a["n_sequences_with_gene"] += 1

    # compute enrichment per family per gene
    summary = {}
    for fam, genes in agg.items():
        summary[fam] = {}
        for gene, a in genes.items():
            if a["n_sequences_with_gene"] < 3: continue
            gene_frac = a["gene_bins_total"] / a["seq_bins_total"]
            expected = a["top_bins_total"] * gene_frac
            obs = a["top_bins_in_gene"]
            oe = obs / expected if expected > 0 else 0.0
            p = float(hypergeom.sf(obs - 1, a["seq_bins_total"],
                                   a["gene_bins_total"], a["top_bins_total"]))
            summary[fam][gene] = {
                "n_seq": a["n_sequences_with_gene"],
                "gene_bins_total": a["gene_bins_total"],
                "top_bins_in_gene": obs,
                "expected": float(expected),
                "observed_expected_ratio": float(oe),
                "hypergeom_p": p,
            }

    out = {"top_frac": TOP_FRAC, "bin_bp": BIN_BP, "summary": summary}
    json.dump(out, open(OUT / "per_gene.json", "w"), indent=2)

    print("\n=== per-gene enrichment (aggregated across sequences per family) ===")
    for fam in summary:
        print(f"\n{fam}:")
        rows = sorted(summary[fam].items(),
                      key=lambda kv: -kv[1]["observed_expected_ratio"])
        for gene, r in rows:
            print(f"  {gene:12s}  n_seq={r['n_seq']:2d}  "
                  f"O/E={r['observed_expected_ratio']:5.2f}  "
                  f"obs={r['top_bins_in_gene']:3d}  "
                  f"exp={r['expected']:5.1f}  "
                  f"p={r['hypergeom_p']:.4g}")


if __name__ == "__main__":
    main()
