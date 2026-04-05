"""
main.py — АгроСкоринг KZ  |  FastAPI Backend
=============================================
Endpoints:
  POST /score  — score a single farm application (called by JS frontend)
  POST /train  — (re-)train the ML model from a CSV upload
  GET  /health — readiness probe

Run:
  pip install fastapi uvicorn python-multipart xgboost shap scikit-learn pandas numpy
  uvicorn main:app --reload --port 8000

CORS is enabled for all origins so the standalone HTML file can fetch
directly from http://localhost:8000 during development.
"""

import io
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our ML engine (same directory)
from ml_engine import (
    FEATURE_COLS,
    MODEL_PATH,
    ScoringInput,
    build_mock_dataset,
    score,
    train,
)

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="АгроСкоринг KZ API",
    description=(
        "Scoring engine for agricultural subsidy applications (Kazakhstan). "
        "Combines rule-based scoring with XGBoost + SHAP explainability. "
        "Final decision is always human-made — the model provides a recommendation only."
    ),
    version="2.0.0",
    docs_url="/docs",
)

# Allow the standalone HTML frontend to call this API without a proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS  (Pydantic)
# ──────────────────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """
    JSON payload sent by the HTML/JS frontend.
    All fields have defaults so partial submissions are accepted gracefully.
    """
    # Identity
    iin:  str = Field("", example="120540001234")
    name: str = Field("Заявитель", example="ИП «Жайлау»")

    # Numeric farm features
    revenue_k_tenge:     float = Field(0,  ge=0, description="Annual revenue, thousands KZT")
    hectares:            float = Field(0,  ge=0, description="Cultivated area, ha")
    years_operating:     int   = Field(0,  ge=0)
    credit_score:        int   = Field(50, ge=0, le=100)
    num_employees:       int   = Field(1,  ge=0)
    equipment_value_m:   float = Field(0,  ge=0, description="Machinery value, millions KZT")
    num_crops:           int   = Field(0,  ge=0)
    num_livestock_types: int   = Field(0,  ge=0)

    # Binary flags
    has_irrigation:  int = Field(0, ge=0, le=1)
    owns_land:       int = Field(1, ge=0, le=1)
    certified_seeds: int = Field(0, ge=0, le=1)

    # Encoded categoricals
    # tax_status_enc:      0=compliant, 1=minor arrears, 2=major arrears
    # subsidy_history_enc: 0=none, 1=once, 2=regular, 3=violations
    # loan_history_enc:    0=none/good, 1=delayed, 2=bad
    tax_status_enc:      int = Field(0, ge=0, le=2)
    subsidy_history_enc: int = Field(0, ge=0, le=3)
    loan_history_enc:    int = Field(0, ge=0, le=2)
    num_documents:       int = Field(0, ge=0, le=6)
    oblast_priority:     int = Field(0, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "iin": "120540001234",
                "name": "ИП «Жайлау»",
                "revenue_k_tenge": 45000,
                "hectares": 250,
                "years_operating": 6,
                "credit_score": 72,
                "num_employees": 12,
                "equipment_value_m": 65,
                "num_crops": 3,
                "num_livestock_types": 1,
                "has_irrigation": 1,
                "owns_land": 1,
                "certified_seeds": 1,
                "tax_status_enc": 0,
                "subsidy_history_enc": 2,
                "loan_history_enc": 0,
                "num_documents": 5,
                "oblast_priority": 1,
            }
        }


class ScoreFactor(BaseModel):
    feature:    str
    name:       str
    shap_value: float
    impact:     float


class ScoreBlock(BaseModel):
    raw: int
    max: int
    pct: int


class ScoreResponse(BaseModel):
    score:        int            = Field(..., description="Hybrid score 0–100")
    rule_score:   int            = Field(..., description="Deterministic rule-based score")
    ml_score:     Optional[int]  = Field(None, description="XGBoost probability × 100")
    engine_used:  str
    verdict:      str            = Field(..., description="approve | review | reject")
    score_blocks: dict[str, ScoreBlock]
    shap_factors: list[ScoreFactor]
    explanation:  str            = Field(..., description="Human-readable XAI explanation in Russian")


class TrainResponse(BaseModel):
    status:   str
    accuracy: Optional[float]
    roc_auc:  Optional[float]
    rows:     int


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: map frontend raw payload → ScoringInput
# ──────────────────────────────────────────────────────────────────────────────

def _map_frontend_payload(req: ScoreRequest) -> ScoringInput:
    """
    The HTML frontend sends some fields in slightly different names (legacy JS).
    This adapter normalises everything so the JS and Python sides can evolve
    independently.
    """
    return ScoringInput(
        iin=req.iin,
        name=req.name,
        revenue_k_tenge=req.revenue_k_tenge,
        hectares=req.hectares,
        years_operating=req.years_operating,
        credit_score=req.credit_score,
        num_employees=req.num_employees,
        equipment_value_m=req.equipment_value_m,
        num_crops=req.num_crops,
        num_livestock_types=req.num_livestock_types,
        has_irrigation=req.has_irrigation,
        owns_land=req.owns_land,
        certified_seeds=req.certified_seeds,
        tax_status_enc=req.tax_status_enc,
        subsidy_history_enc=req.subsidy_history_enc,
        loan_history_enc=req.loan_history_enc,
        num_documents=req.num_documents,
        oblast_priority=req.oblast_priority,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Readiness probe — also reports model status."""
    return {
        "status": "ok",
        "model_trained": os.path.exists(MODEL_PATH),
        "version": "2.0.0",
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_application(req: ScoreRequest):
    """
    Score a single farm subsidy application.

    The engine:
    1. Runs the deterministic 5-block rule engine (always).
    2. Runs the XGBoost classifier if model.pkl exists.
    3. Blends both into a hybrid score (40 % rules + 60 % ML).
    4. Generates SHAP-based per-feature explanations.
    5. Returns a human-readable recommendation in Russian.

    **The returned verdict is a recommendation only.**
    The final decision must be made by an authorised human expert.
    """
    try:
        inp    = _map_frontend_payload(req)
        result = score(inp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Coerce score_blocks to ScoreBlock models
    coerced_blocks = {
        k: ScoreBlock(**v) for k, v in result["score_blocks"].items()
    }
    # Coerce shap_factors
    coerced_factors = [ScoreFactor(**f) for f in result["shap_factors"]]

    return ScoreResponse(
        score=result["score"],
        rule_score=result["rule_score"],
        ml_score=result["ml_score"],
        engine_used=result["engine_used"],
        verdict=result["verdict"],
        score_blocks=coerced_blocks,
        shap_factors=coerced_factors,
        explanation=result["explanation"],
    )


@app.post("/train", response_model=TrainResponse, tags=["Model Management"])
async def train_model(file: Optional[UploadFile] = File(None)):
    """
    Train (or re-train) the XGBoost model.

    - If a CSV file is uploaded, use it as the training dataset.
    - If no file is provided, train on the built-in mock dataset (useful for demos).

    Expected CSV columns: all items in `FEATURE_COLS` + `approved` (0 or 1).

    This endpoint is intended for authorised administrators only —
    add auth middleware (e.g. OAuth2 / API key) in production.
    """
    df = None
    rows = 0

    if file is not None:
        try:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
            # Validate required columns
            missing = set(FEATURE_COLS + ["approved"]) - set(df.columns)
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"CSV is missing required columns: {sorted(missing)}",
                )
            df = df[FEATURE_COLS + ["approved"]].dropna()
            rows = len(df)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    else:
        df   = build_mock_dataset(n=300)
        rows = len(df)

    try:
        metrics = train(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    return TrainResponse(
        status="Model trained and saved successfully.",
        accuracy=metrics["accuracy"],
        roc_auc=metrics["roc_auc"],
        rows=rows,
    )
