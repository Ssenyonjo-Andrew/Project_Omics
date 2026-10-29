"""Request/response contracts."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SequenceInput(BaseModel):
    sequence: str = Field(..., min_length=50, max_length=250_000)
    name: Optional[str] = None

    @field_validator("sequence")
    @classmethod
    def clean_sequence(cls, v: str) -> str:
        lines = [l.strip() for l in v.strip().splitlines() if not l.startswith(">")]
        s = "".join(lines).upper().replace("U", "T")
        if not s:
            raise ValueError("sequence is empty")
        bad = sum(c not in "ACGTN" for c in s)
        if bad > len(s) * 0.1:
            raise ValueError(f"sequence contains {bad} non-ACGT characters")
        return s


class Prediction(BaseModel):
    virus: str
    probs: list[float]
    confidence: float


class PredictionResponse(BaseModel):
    sequence_length: int
    cnn: Prediction
    bilstm: Prediction


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
