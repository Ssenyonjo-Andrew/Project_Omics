import json, uuid, logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from app import inference
from app.schemas import SequenceInput  # <--- FIX: Import the strict schema

logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).parent
app = FastAPI(title="xAI Clinical Dashboard")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

JOBS = {}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/benchmark", response_class=HTMLResponse)
def benchmark(request: Request):
    return templates.TemplateResponse(request, "benchmark.html", {})

@app.post("/api/attribute")
def attr(payload: SequenceInput, bg: BackgroundTasks): # <--- FIX: Use strict schema
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "step": "starting", "predictions": None, "metrics": None, "error": None}
    
    def run():
        try:
            seq = payload.sequence # Already cleaned by the schema (no > headers, no U)
            JOBS[job_id]["step"] = "predicting"
            preds = inference.predict_all(seq)
            JOBS[job_id]["predictions"] = preds
            
            JOBS[job_id]["step"] = "metrics"
            top_v = preds["transformer"]["virus"]
            metrics = inference.calculate_xai_metrics(seq, top_v)
            JOBS[job_id]["metrics"] = metrics
            
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["step"] = "done"
        except Exception as e:
            logging.error(f"Job {job_id} failed: {str(e)}")
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
        
    bg.add_task(run)
    return {"job_id": job_id}

@app.get("/api/attribute/{job_id}")
def attr_status(job_id: str):
    if job_id not in JOBS: raise HTTPException(404)
    return JOBS[job_id]
