import json, uuid, logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from app import inference

logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).parent
app = FastAPI(title="xAI Clinical Dashboard")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

JOBS = {}

class SeqInput(BaseModel):
    sequence: str = Field(..., min_length=50)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/api/attribute")
def attr(payload: SeqInput, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None}
    
    def run():
        JOBS[job_id]["status"] = "running"
        seq = "".join(payload.sequence.split()).upper()
        res = inference.calculate_all_metrics(seq)
        JOBS[job_id]["result"] = res
        JOBS[job_id]["status"] = "done"
        
    bg.add_task(run)
    return {"job_id": job_id}

@app.get("/api/attribute/{job_id}")
def attr_status(job_id: str):
    if job_id not in JOBS: raise HTTPException(404)
    return JOBS[job_id]
