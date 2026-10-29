"""FastAPI application."""
import json, uuid, logging, os
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.schemas import SequenceInput, PredictionResponse, JobStatus
from app import inference

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("xai-app")

APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parent.parent  # /fyp
RESULTS = ROOT / "results"

app = FastAPI(title="xAI Viral Attribution", version="1.0.0",
              description="Stability, fidelity and plausibility of feature attributions in viral classifiers")

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

# In-memory job store (single-user; replace with Redis for multi-user)
JOBS: dict = {}


def _load_bundle():
    p = RESULTS / "prototype" / "bundle.json"
    if not p.exists():
        raise RuntimeError(f"missing {p}; run 09_build_bundle.py first")
    return json.load(open(p))


def _load_eval():
    return json.load(open(RESULTS / "eval_v2" / "evaluation.json"))


def _load_train():
    return json.load(open(RESULTS / "train_results.json"))


BUNDLE = _load_bundle()
EVAL = _load_eval()
TRAIN = _load_train()


# =====================================================================
# Pages
# =====================================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    import numpy as np
    stab = EVAL.get("stability_summary", {})
    medians = [v["median"] for v in stab.values() if isinstance(v, dict)]
    return templates.TemplateResponse(request, "index.html", {
        "kpis": {
            "cnn_f1": TRAIN["cnn"]["test"]["macro_f1"],
            "bilstm_f1": TRAIN["bilstm"]["test"]["macro_f1"],
            "chance": 0.25,
            "stability": float(np.median(medians)) if medians else 0.0,
        },
        "n_test": len(BUNDLE["sequences"]),
        "n_viruses": len(BUNDLE["viruses"]),
    })


@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request):
    return templates.TemplateResponse(request, "analyze.html", {})


@app.get("/explore", response_class=HTMLResponse)
def explore_page(request: Request):
    seqs = BUNDLE["sequences"]
    rows = []
    for acc, s in seqs.items():
        rows.append({
            "accession": acc,
            "virus": s["virus"],
            "length": s["length"],
            "cnn_pred": s["cnn_pred"],
            "cnn_correct": s["cnn_pred"] == s["virus"],
            "bilstm_pred": s["bilstm_pred"],
            "bilstm_correct": s["bilstm_pred"] == s["virus"],
            "stability": s.get("stability"),
        })
    rows.sort(key=lambda r: (r["virus"], r["accession"]))
    return templates.TemplateResponse(request, "explore.html", {
        "request": request, "rows": rows,
    })


@app.get("/study", response_class=HTMLResponse)
def study_page(request: Request):
    return templates.TemplateResponse(request, "study.html", {
        "train": TRAIN,
        "stability": EVAL.get("stability_summary", {}),
        "fidelity": _aggregate_fidelity(),
        "plausibility": _aggregate_plausibility(),
    })


# =====================================================================
# API
# =====================================================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/predict", response_model=PredictionResponse)
def api_predict(payload: SequenceInput):
    seq = payload.sequence
    log.info(f"predict: {len(seq)} bp")
    cnn = inference.predict(seq, "cnn")
    bil = inference.predict(seq, "bilstm")
    return PredictionResponse(
        sequence_length=len(seq),
        cnn=cnn,
        bilstm=bil,
    )


def _run_attribution(job_id: str, seq: str, accession: str | None):
    JOBS[job_id]["status"] = "running"
    try:
        JOBS[job_id]["progress"] = "cnn"
        cnn_attr = inference.attribute(seq, "cnn")
        JOBS[job_id]["progress"] = "bilstm"
        bil_attr = inference.attribute(seq, "bilstm")
        genes = inference.fetch_genes(accession) if accession else {}
        JOBS[job_id]["result"] = {
            "cnn": cnn_attr,
            "bilstm": bil_attr,
            "genes": genes,
            "sequence_length": len(seq),
        }
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = "complete"
    except Exception as e:
        log.exception("attribution failed")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.post("/api/attribute")
def api_attribute(payload: SequenceInput, background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": "queued",
                    "result": None, "error": None}
    background.add_task(_run_attribution, job_id, payload.sequence, payload.name)
    log.info(f"attribution job queued: {job_id}")
    return {"job_id": job_id}


@app.get("/api/attribute/{job_id}", response_model=JobStatus)
def api_attribute_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job not found")
    j = JOBS[job_id]
    return JobStatus(job_id=job_id, status=j["status"],
                     progress=j["progress"], result=j["result"],
                     error=j["error"])



@app.post("/api/fidelity")
def api_fidelity(payload: dict):
    seq = payload["sequence"]
    model = payload["model"]
    attr_dict = payload["attribution"]
    result = inference.calculate_fidelity(seq, model, attr_dict)
    return result


@app.get("/api/sequences")
def api_sequences():
    return [{"accession": a, **s} for a, s in BUNDLE["sequences"].items()]


@app.get("/api/sequences/{accession}")
def api_sequence(accession: str):
    if accession not in BUNDLE["sequences"]:
        raise HTTPException(status_code=404, detail="sequence not found")
    return {"accession": accession, **BUNDLE["sequences"][accession]}


# =====================================================================
# Helpers
# =====================================================================
def _aggregate_fidelity():
    import numpy as np
    out = defaultdict(dict)
    for rec in EVAL["fidelity"].values():
        v = rec["virus"]
        for frac in ["0.01", "0.05", "0.1", "0.2"]:
            out[v].setdefault(frac, {"cnn": [], "bilstm": [], "rand_cnn": [], "rand_bilstm": []})
            out[v][frac]["cnn"].append(rec["deletion"][frac]["cnn"])
            out[v][frac]["bilstm"].append(rec["deletion"][frac]["bilstm"])
            out[v][frac]["rand_cnn"].append(rec["deletion"][frac]["random_cnn"])
            out[v][frac]["rand_bilstm"].append(rec["deletion"][frac]["random_bilstm"])
    return {v: {f: {k: float(np.median(x)) for k, x in d.items()} for f, d in fv.items()}
            for v, fv in out.items()}


def _aggregate_plausibility():
    import numpy as np
    out = defaultdict(lambda: {"antigenic": [], "conserved": []})
    for rec in EVAL["plausibility"].values():
        if rec.get("antigenic_oe") is not None:
            out[rec["virus"]]["antigenic"].append(rec["antigenic_oe"])
        if rec.get("conserved_oe") is not None:
            out[rec["virus"]]["conserved"].append(rec["conserved_oe"])
    return {v: {"n_antigenic": len(d["antigenic"]),
                "median_antigenic": float(np.median(d["antigenic"])) if d["antigenic"] else None,
                "n_conserved": len(d["conserved"]),
                "median_conserved": float(np.median(d["conserved"])) if d["conserved"] else None}
            for v, d in out.items()}
