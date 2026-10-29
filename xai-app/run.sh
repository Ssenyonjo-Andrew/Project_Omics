#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Starting xAI Viral Attribution"
echo "  API docs: http://localhost:8000/docs"
echo "  App:      http://localhost:8000/"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
