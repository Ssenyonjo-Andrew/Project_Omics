# xAI Viral Attribution

Production-grade web tool for live viral sequence classification and feature attribution.

## Stack
- FastAPI (async framework, OpenAPI spec)
- Jinja2 (server-rendered templates)
- Pydantic v2 (request/response contracts)
- PyTorch (inference)
- Plotly.js (client-side charts)

## Setup
    cd xai-app
    pip install -r requirements.txt

## Run
    ./run.sh

Then:
- Application: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | / | Landing page |
| GET | /analyze | Upload + live inference UI |
| GET | /explore | Test set browser |
| GET | /study | Findings and aggregate results |
| POST | /api/predict | JSON: sequence -> CNN + BiLSTM |
| POST | /api/attribute | JSON: sequence -> job_id |
| GET | /api/attribute/{job_id} | Poll job status |
| GET | /api/sequences | List pre-computed sequences |
| GET | /health | Health check |
