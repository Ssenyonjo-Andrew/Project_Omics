"""Professional Streamlit prototype: xAI for viral family classification."""
import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio import SeqIO

st.set_page_config(
    page_title="xAI Viral Attribution",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CFG = yaml.safe_load(open("config.yaml"))
FAMILIES = CFG["models"]["families"]
FAM2IDX = {f: i for i, f in enumerate(FAMILIES)}
N = len(FAMILIES)
CKPT = Path(CFG["training"]["checkpoint_dir"])
BIN_BP = 100

# Vivid palette
FAMILY_COLOR = {
    "Coronaviridae":   "#6366F1",  # indigo
    "Filoviridae":     "#EF4444",  # red
    "Flaviviridae":    "#10B981",  # emerald
    "Arenaviridae":    "#A855F7",  # violet
    "Paramyxoviridae": "#F97316",  # orange
}
MODEL_COLOR = {"cnn": "#6366F1", "bilstm": "#EC4899"}
ACCENT = "#8B5CF6"


# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: linear-gradient(160deg, #F8FAFC 0%, #EEF2FF 55%, #FCE7F3 100%); }
.block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3, h4 { color: #1E1B4B !important; font-weight: 800 !important; letter-spacing: -0.02em; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 45%, #EC4899 100%);
    padding: 2.4rem 2.6rem;
    border-radius: 22px;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 24px 60px -12px rgba(99, 102, 241, 0.45);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
}
.hero h1 {
    color: white !important; font-size: 2.1rem !important;
    margin: 0 0 0.4rem 0; font-weight: 800;
    display: flex; align-items: center; gap: 0.7rem;
}
.hero p { color: rgba(255,255,255,0.92); margin: 0; font-size: 1.02rem; font-weight: 500; }
.hero .tag {
    display: inline-block; margin-top: 1rem;
    padding: 0.28rem 0.85rem; border-radius: 999px;
    background: rgba(255,255,255,0.18); color: white;
    font-size: 0.82rem; font-weight: 600;
    backdrop-filter: blur(10px);
    margin-right: 0.4rem;
}

/* Metric cards */
.kpi { display: flex; gap: 1rem; margin-bottom: 1.8rem; }
.kpi-card {
    flex: 1; padding: 1.2rem 1.4rem; border-radius: 16px;
    background: white; border: 1px solid #E0E7FF;
    box-shadow: 0 4px 20px -8px rgba(99, 102, 241, 0.18);
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366F1, #EC4899);
}
.kpi-label { font-size: 0.78rem; color: #64748B; text-transform: uppercase;
             letter-spacing: 0.08em; font-weight: 600; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #1E1B4B;
             margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace; }
.kpi-sub { font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem; }

/* Section headers */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #1E1B4B;
    margin: 1.8rem 0 0.7rem 0; display: flex; align-items: center; gap: 0.6rem;
}
.section-title .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, #6366F1, #EC4899); color: white;
    font-size: 0.85rem; font-weight: 700;
}

/* Predictions cards */
.pred-card {
    padding: 1.3rem 1.5rem; border-radius: 16px;
    background: white; border: 2px solid transparent;
    box-shadow: 0 4px 20px -8px rgba(0,0,0,0.08);
    margin-bottom: 0.6rem;
}
.pred-card.cnn { border-color: #6366F1; }
.pred-card.bilstm { border-color: #EC4899; }
.pred-label { font-size: 0.8rem; font-weight: 700; color: #64748B;
              text-transform: uppercase; letter-spacing: 0.1em; }
.pred-class { font-size: 1.5rem; font-weight: 800; margin: 0.3rem 0;
              font-family: 'JetBrains Mono', monospace; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem; background: white; padding: 0.4rem;
    border-radius: 14px; border: 1px solid #E0E7FF;
    box-shadow: 0 4px 20px -8px rgba(99, 102, 241, 0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 0.65rem 1.3rem;
    color: #64748B; font-weight: 600; font-size: 0.92rem;
    border: none; background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #EC4899);
    color: white; border: none; padding: 0.7rem 1.6rem;
    font-weight: 700; border-radius: 12px;
    box-shadow: 0 8px 24px -8px rgba(99, 102, 241, 0.6);
    transition: all 0.2s; letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -8px rgba(99, 102, 241, 0.8);
}

/* Inputs */
.stTextArea textarea, .stFileUploader, .stSelectbox > div > div, .stRadio > div {
    border-radius: 12px !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Success / info boxes */
.stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ---------------- models ----------------
class CNN(nn.Module):
    def __init__(self, f1, f2, k, dr):
        super().__init__()
        self.c1 = nn.Conv1d(4, f1, k, padding="same"); self.b1 = nn.BatchNorm1d(f1)
        self.c2 = nn.Conv1d(f1, f2, k, padding="same"); self.b2 = nn.BatchNorm1d(f2)
        self.pool = nn.AdaptiveMaxPool1d(1); self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(f2, N)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x))); x = F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))

class BiLSTM(nn.Module):
    def __init__(self, v, e, h, dr):
        super().__init__()
        self.emb = nn.Embedding(v, e, padding_idx=0)
        self.lstm = nn.LSTM(e, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dr); self.fc = nn.Linear(h * 2, N)
    def forward(self, x):
        out, _ = self.lstm(self.emb(x)); return self.fc(self.drop(out.mean(1)))

@st.cache_resource
def load_models():
    c = CFG["models"]["cnn"]; b = CFG["models"]["bilstm"]
    cnn = CNN(c["filters"][0], c["filters"][1], c["kernel"], c["dropout"])
    cnn.load_state_dict(torch.load(CKPT / "cnn.pt", map_location="cpu")); cnn.eval()
    bil = BiLSTM(b["vocab"], b["embed"], b["hidden"], b["dropout"])
    bil.load_state_dict(torch.load(CKPT / "bilstm.pt", map_location="cpu")); bil.eval()
    return cnn, bil, c, b

@st.cache_data
def load_demo():
    return json.load(open("results/prototype_data.json"))


# ---------------- encoding ----------------
def windows(seq, wlen, stride):
    if len(seq) < wlen: yield seq + "N" * (wlen - len(seq)); return
    for i in range(0, len(seq) - wlen + 1, stride): yield seq[i:i + wlen]
    if (len(seq) - wlen) % stride: yield seq[-wlen:]

DNA = {"A":0,"C":1,"G":2,"T":3}
def onehot(s):
    x = np.zeros((4, len(s)), dtype=np.float32)
    for i, c in enumerate(s):
        if c in DNA: x[DNA[c], i] = 1.0
    return x

def kmers(s, k):
    ids = []
    for i in range(0, len(s) - k + 1, k):
        idx, ok = 0, True
        for c in s[i:i + k]:
            if c not in "ACGT": ok = False; break
            idx = idx * 4 + "ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]

@torch.no_grad()
def predict(model, seqs, window, stride, k=None, is_lstm=False, batch=32):
    ws, owner = [], []
    for i, s in enumerate(seqs):
        for w in windows(s, window, stride):
            ws.append(kmers(w, k) if is_lstm else onehot(w))
            owner.append(i)
    owner = np.array(owner)
    if is_lstm:
        L = max(len(t) for t in ws); X = np.zeros((len(ws), L), dtype=np.int64)
        for i, t in enumerate(ws): X[i, :len(t)] = t
        X = torch.from_numpy(X)
    else:
        X = torch.from_numpy(np.stack(ws))
    probs = []
    for j in range(0, len(X), batch):
        probs.append(F.softmax(model(X[j:j + batch]), -1).numpy())
    probs = np.concatenate(probs)
    out = np.zeros((len(seqs), N))
    for i in range(len(seqs)): out[i] = probs[owner == i].mean(0)
    return out


# ---------------- attribution ----------------
def occlude_bins(seq, bin_bp=BIN_BP):
    n = (len(seq) + bin_bp - 1) // bin_bp
    return [seq[:b*bin_bp] + "N"*min(bin_bp, len(seq)-b*bin_bp) + seq[(b+1)*bin_bp:]
            for b in range(n)], n

def ig_cnn(model, seq, fam_idx, window, stride, steps=16):
    L = len(seq); n_bins = (L + BIN_BP - 1) // BIN_BP
    attr = np.zeros(n_bins)
    starts = list(range(0, max(L-window,0)+1, stride))
    if not starts or starts[-1] != L-window: starts.append(max(L-window, 0))
    for s in starts:
        win = seq[s:s+window]
        if len(win) < window: win += "N"*(window-len(win))
        x = torch.from_numpy(onehot(win))
        b = torch.full_like(x, 0.25); d = x - b
        alphas = torch.linspace(0, 1, steps).reshape(-1,1,1)
        interp = (b + alphas*d).detach().requires_grad_(True)
        out = model(interp)
        grads = torch.autograd.grad(out[:, fam_idx].sum(), interp)[0]
        per_pos = (grads.mean(0) * d).sum(0).abs().numpy()
        for p, v in enumerate(per_pos):
            if s+p >= L: break
            attr[(s+p)//BIN_BP] += v
    return attr


@st.cache_data(show_spinner=False)
def fetch_genes(accession):
    import subprocess, tempfile
    cache = Path(f"data/raw/genbank/{accession}.gb")
    path, cleanup = None, False
    if cache.exists() and cache.stat().st_size > 0:
        path = str(cache)
    else:
        try:
            r = subprocess.run(["efetch","-db","nuccore","-id",accession,"-format","gb"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0 or not r.stdout.strip():
                return {}
            tf = tempfile.NamedTemporaryFile(mode="w+", suffix=".gb", delete=False)
            tf.write(r.stdout); tf.close()
            path, cleanup = tf.name, True
        except Exception:
            return {}
    genes = {}
    try:
        for rec in SeqIO.parse(path, "genbank"):
            for feat in rec.features:
                if feat.type not in ("CDS","mat_peptide","gene"): continue
                name = None
                for key in ("gene","product","standard_name"):
                    if key in feat.qualifiers:
                        name = feat.qualifiers[key][0]; break
                if not name: continue
                name = name.upper().strip()
                for suf in (" PROTEIN"," POLYPROTEIN"," (PP)"):
                    if name.endswith(suf): name = name[:-len(suf)].strip()
                try:
                    genes.setdefault(name, []).append(
                        [int(feat.location.start), int(feat.location.end)])
                except Exception:
                    for part in feat.location.parts:
                        genes.setdefault(name, []).append(
                            [int(part.start), int(part.end)])
    finally:
        if cleanup: Path(path).unlink(missing_ok=True)
    return genes


# ---------------- hero ----------------
st.markdown("""
<div class="hero">
    <h1>🧬 xAI Viral Attribution</h1>
    <p>Upload a viral nucleotide sequence → get a family prediction and a live attribution track</p>
    <div style="margin-top: 1rem;">
        <span class="tag">CNN + BiLSTM</span>
        <span class="tag">Occlusion · Integrated Gradients</span>
        <span class="tag">Gene-aware attribution</span>
        <span class="tag">CPU inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

cnn, bil, c, b = load_models()
D = load_demo()
H = D["headline"]

st.markdown(f"""
<div class="kpi">
  <div class="kpi-card">
    <div class="kpi-label">CNN · test macro-F1</div>
    <div class="kpi-value">{H['cnn_test_f1']:.3f}</div>
    <div class="kpi-sub">3-way, species held out</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">BiLSTM · test macro-F1</div>
    <div class="kpi-value">{H['bilstm_test_f1']:.3f}</div>
    <div class="kpi-sub">3-way, species held out</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Chance baseline</div>
    <div class="kpi-value">{H['chance_3way']:.3f}</div>
    <div class="kpi-sub">uniform over 3 families</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Attribution stability (ρ)</div>
    <div class="kpi-value">{H['stability_median']:.2f}</div>
    <div class="kpi-sub">CNN vs BiLSTM consensus</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_run, tab_eval, tab_about = st.tabs(
    ["🔬  Analyse a sequence", "📊  System evaluation", "📖  Methodology"]
)

# ============================================================
# TAB 1
# ============================================================
with tab_run:
    st.markdown('<div class="section-title"><span class="num">1</span>Provide a sequence</div>',
                unsafe_allow_html=True)

    mode = st.radio(
        "Input method",
        ["Paste FASTA", "Upload file", "Known test accession"],
        horizontal=True, label_visibility="collapsed",
    )

    seq, acc_hint, source = None, None, None

    if mode == "Paste FASTA":
        pasted = st.text_area("Paste FASTA", height=160, label_visibility="collapsed",
                              placeholder=">my_sequence\nATCGTACG...")
        if pasted.strip():
            if pasted.startswith(">"):
                lines = pasted.strip().splitlines()
                acc_hint = lines[0][1:].split()[0]
                seq = "".join(l.strip() for l in lines[1:])
            else:
                seq = "".join(pasted.split())
            source = "pasted"

    elif mode == "Upload file":
        up = st.file_uploader("FASTA file", type=["fa","fasta","fna","txt"])
        if up is not None:
            content = up.read().decode("utf-8", errors="ignore")
            if content.startswith(">"):
                lines = content.strip().splitlines()
                acc_hint = lines[0][1:].split()[0]
                seq = "".join(l.strip() for l in lines[1:])
            else:
                seq = "".join(content.split())
            source = up.name

    else:
        accs = sorted(D["sequences"].keys())
        acc = st.selectbox("Accession", accs, index=0)
        acc_hint = acc
        df = pd.read_csv("data/processed/master_final.csv")
        row = df[df["accession"] == acc].iloc[0]
        for rec in SeqIO.parse(row["sequence_file_path"], "fasta"):
            if rec.id.split(".")[0] == acc.split(".")[0]:
                seq = str(rec.seq).upper(); break
        source = f"test set · {D['sequences'][acc]['family']}"

    if not seq:
        st.info("Provide a sequence above to run the analysis.")
        st.stop()

    seq = seq.upper().replace("U", "T")
    bad = sum(ch not in "ACGTN" for ch in seq)
    if bad > len(seq) * 0.1:
        st.error(f"Sequence has {bad} non-ACGT characters — is this a nucleotide sequence?")
        st.stop()

    L = len(seq)
    st.success(f"**{L:,} bp** · {bad} ambiguous positions · source: {source}")

    st.markdown('<div class="section-title"><span class="num">2</span>Family prediction</div>',
                unsafe_allow_html=True)

    with st.spinner("Running both models…"):
        p_cnn = predict(cnn, [seq], c["window"], c["stride"])[0]
        p_bil = predict(bil, [seq], b["window"], b["stride"], b["kmer"], is_lstm=True)[0]

    col_cnn, col_bil = st.columns(2)
    for col, label, probs, key, color in [
        (col_cnn, "CNN",      p_cnn, "cnn",    MODEL_COLOR["cnn"]),
        (col_bil, "BiLSTM",   p_bil, "bilstm", MODEL_COLOR["bilstm"]),
    ]:
        with col:
            pred = FAMILIES[int(np.argmax(probs))]
            pred_color = FAMILY_COLOR[pred]
            st.markdown(f"""
            <div class="pred-card {key}">
                <div class="pred-label">{label}</div>
                <div class="pred-class" style="color: {pred_color};">{pred}</div>
                <div style="font-size:0.82rem;color:#64748B;">
                    confidence {probs.max():.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=probs, y=FAMILIES, orientation="h",
                marker_color=[FAMILY_COLOR[f] for f in FAMILIES],
                marker_line_width=0,
                text=[f"{p:.2f}" for p in probs], textposition="outside",
                textfont=dict(size=11, color="#1E1B4B"),
            ))
            fig.update_layout(
                height=230, margin=dict(l=10, r=30, t=10, b=10),
                xaxis_range=[0, 1.05], showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#E5E7EB", zeroline=False,
                           showline=False, ticks=""),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#1E1B4B")),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"pred_{key}")

    st.markdown('<div class="section-title"><span class="num">3</span>Attribution</div>',
                unsafe_allow_html=True)
    st.caption("Occlusion + Integrated Gradients on the CNN. Runs in ~10–15 s on CPU. "
               "Full 8-method consensus for the test set is in the System evaluation tab.")

    if st.button("⚡  Run attribution", type="primary"):
        fidx = int(np.argmax(p_cnn))
        with st.spinner("Computing occlusion…"):
            occl_seqs, n_bins = occlude_bins(seq)
            base = p_cnn[fidx]
            occl = predict(cnn, occl_seqs, c["window"], c["stride"])[:, fidx]
            occl_attr = base - occl
        with st.spinner("Computing Integrated Gradients…"):
            ig_attr = ig_cnn(cnn, seq, fidx, c["window"], c["stride"], steps=16)

        consensus = (np.abs(occl_attr) + np.abs(ig_attr)) / 2
        cons_norm = consensus / consensus.max() if consensus.max() > 0 else consensus

        genes = {}
        if acc_hint:
            with st.spinner(f"Fetching gene annotation for {acc_hint}…"):
                genes = fetch_genes(acc_hint)

        st.markdown(
            f"<div style='padding:1rem 1.2rem;background:white;border-radius:14px;"
            f"border-left:4px solid {FAMILY_COLOR[FAMILIES[fidx]]};"
            f"box-shadow:0 4px 20px -8px rgba(0,0,0,0.08);margin-bottom:1rem;'>"
            f"<b style='color:#64748B;font-size:0.8rem;text-transform:uppercase;"
            f"letter-spacing:0.08em;'>Attributed class</b><br>"
            f"<span style='font-size:1.4rem;font-weight:800;"
            f"color:{FAMILY_COLOR[FAMILIES[fidx]]};'>{FAMILIES[fidx]}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        positions_kb = np.arange(n_bins) * BIN_BP / 1000
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=positions_kb, y=cons_norm,
            marker=dict(color=cons_norm, colorscale="Plasma",
                        line=dict(width=0)),
            name="attribution", hovertemplate="pos %{x:.1f} kb<br>importance %{y:.2f}<extra></extra>",
        ))

        for gene, spans in genes.items():
            for j, (st_, en) in enumerate(spans):
                fig.add_vrect(
                    x0=st_/1000, x1=en/1000,
                    fillcolor="rgba(99, 102, 241, 0.07)",
                    line_width=0,
                    annotation_text=gene if j == 0 else "",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="#6366F1",
                )

        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="genome position (kb)", gridcolor="#E5E7EB",
                       showline=False),
            yaxis=dict(title="normalized importance", gridcolor="#E5E7EB",
                       showline=False),
            showlegend=False, bargap=0,
        )
        st.plotly_chart(fig, use_container_width=True, key="attr_track")

        if not genes:
            st.caption("No gene annotation — NCBI could not resolve this accession. "
                       "Attribution itself is unaffected.")

        st.markdown("**Top 10 attributed regions**")
        top = np.argsort(-consensus)[:10]
        rows = []
        for rank, b_idx in enumerate(top, 1):
            start = b_idx * BIN_BP
            end = min(start + BIN_BP, L)
            rows.append({
                "Rank": rank,
                "Position": f"{start:,}–{end:,} bp",
                "Attribution": f"{consensus[b_idx]:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ============================================================
# TAB 2
# ============================================================
with tab_eval:
    st.markdown('<div class="section-title"><span class="num">1</span>Classifier performance</div>',
                unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=["Chance","BiLSTM","CNN"],
        y=[H["chance_3way"], H["bilstm_test_f1"], H["cnn_test_f1"]],
        marker=dict(color=["#94A3B8", MODEL_COLOR["bilstm"], MODEL_COLOR["cnn"]],
                    line=dict(width=0)),
        text=[f"{v:.3f}" for v in [H["chance_3way"], H["bilstm_test_f1"], H["cnn_test_f1"]]],
        textposition="outside", textfont=dict(size=13, color="#1E1B4B"),
    ))
    fig.update_layout(
        height=300, yaxis_range=[0, 1.1],
        yaxis_title="macro-F1 (3-way, species held out)",
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title"><span class="num">2</span>Attribution stability</div>',
                unsafe_allow_html=True)
    st.metric("Median Spearman · CNN vs BiLSTM", f"{H['stability_median']:.2f}",
              help="Across 45 held-out test sequences")
    st.caption("Low agreement means the two architectures identify different genomic "
               "regions as important — the central finding of §5.3.")

    st.markdown('<div class="section-title"><span class="num">3</span>Fidelity (margin deletion)</div>',
                unsafe_allow_html=True)
    fid = D["fidelity"]; fracs = ["0.01","0.05","0.1","0.2"]
    rows = []
    for fam in H["families"]:
        sub = [v for v in fid.values() if v["family"] == fam]
        if not sub: continue
        for mk, ml in [("cnn","CNN"),("bilstm","BiLSTM")]:
            for frac in fracs:
                d = [r["deletion"][frac][f"drop_{mk}"] for r in sub]
                rows.append({"family": fam, "model": ml,
                             "deleted": f"{float(frac)*100:.0f}%",
                             "margin_drop": float(np.median(d))})
    fig = px.bar(pd.DataFrame(rows), x="deleted", y="margin_drop",
                 color="model", facet_col="family", barmode="group",
                 color_discrete_map={"CNN": MODEL_COLOR["cnn"],
                                     "BiLSTM": MODEL_COLOR["bilstm"]},
                 labels={"margin_drop":"margin drop"})
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title"><span class="num">4</span>Per-gene plausibility</div>',
                unsafe_allow_html=True)
    pg = D["per_gene"]
    rows = [{"family": fam, "gene": gene, "n": r["n_seq"],
             "O/E": r["observed_expected_ratio"], "p": r["hypergeom_p"]}
            for fam, gs in pg.items() for gene, r in gs.items()]
    pgdf = pd.DataFrame(rows).sort_values(["family","O/E"], ascending=[True,False])
    for fam in pgdf["family"].unique():
        st.markdown(f"**{fam}**")
        sub = pgdf[pgdf["family"]==fam]
        fig = go.Figure(go.Bar(
            x=sub["O/E"], y=sub["gene"], orientation="h",
            marker=dict(
                color=["#10B981" if oe>1.3 else "#EF4444" if oe<0.7 else "#94A3B8"
                       for oe in sub["O/E"]],
                line=dict(width=0),
            ),
            text=[f"{oe:.2f} (p={p:.2g})" for oe,p in zip(sub["O/E"],sub["p"])],
            textposition="outside", textfont=dict(size=11, color="#1E1B4B"),
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="#64748B", line_width=1)
        fig.update_layout(
            height=max(230, 32*len(sub)), margin=dict(l=10, r=100, t=10, b=10),
            xaxis_title="observed / expected", yaxis_title="", showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#E5E7EB"),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#1E1B4B")),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"pg_{fam}")

# ============================================================
# TAB 3
# ============================================================
with tab_about:
    st.markdown("""
    <div style="background:white;border-radius:16px;padding:2rem;
                box-shadow:0 4px 20px -8px rgba(0,0,0,0.08);">
    <h3 style="margin-top:0;">What this prototype demonstrates</h3>
    <p style="color:#475569;line-height:1.7;">
    The communication layer of a study evaluating <b>stability</b>,
    <b>fidelity</b> and <b>biological plausibility</b> of feature attributions
    in a deep learning viral family classifier.
    </p>

    <h4 style="color:#6366F1;margin-top:1.5rem;">Live analysis (Tab 1)</h4>
    <ul style="color:#475569;line-height:1.8;">
        <li><b>Prediction</b> runs both architectures live on the uploaded sequence.</li>
        <li><b>Attribution</b> runs two of the four methods (occlusion and
            Integrated Gradients on the CNN) in ~10–15 s on CPU.</li>
        <li><b>Gene overlay</b> is fetched from NCBI GenBank when the FASTA
            header contains an accession NCBI recognises.</li>
    </ul>

    <h4 style="color:#EC4899;margin-top:1.5rem;">Reference results (Tab 2)</h4>
    <ul style="color:#475569;line-height:1.8;">
        <li>Full 8-method consensus ranking on 45 held-out test sequences.</li>
        <li>Stability, fidelity, and per-gene plausibility from the study's evaluation.</li>
    </ul>

    <h4 style="color:#F97316;margin-top:1.5rem;">Not shown here</h4>
    <ul style="color:#475569;line-height:1.8;">
        <li>BiLSTM attribution (~30 s), KernelSHAP (~50 s) — too slow for interactive use.</li>
        <li>Retraining-without-top-features — computationally prohibitive here.</li>
    </ul>

    <h4 style="color:#10B981;margin-top:1.5rem;">Model</h4>
    <ul style="color:#475569;line-height:1.8;">
        <li>CNN and BiLSTM trained on <b>783 sequences</b> across 9 species and 5 families.</li>
        <li>Test: 3 species held out entirely (MERS-CoV, Ebola, Dengue).</li>
        <li>Test macro-F1: CNN 0.749 · BiLSTM 0.565 · chance 0.333.</li>
    </ul>

    <p style="color:#94A3B8;font-size:0.85rem;margin-top:2rem;border-top:1px solid #E5E7EB;padding-top:1rem;">
    Full methodology in Chapter 3 · results in Chapter 5 · interpretation in Chapter 6.
    </p>
    </div>
    """, unsafe_allow_html=True)
