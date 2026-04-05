"""
ml_engine.py — АгроСкоринг KZ  |  Minimum Working Element (Stage 1)
=====================================================================
Architecture:
  1. Feature Engineering  — raw input → normalised numeric features
  2. Rule Engine          — deterministic 5-block weighted score (fallback)
  3. XGBoost Classifier   — probability of subsidy approval (0-100)
  4. Hybrid Combiner      — 0.4 × rule_score + 0.6 × ml_score
  5. SHAP Explainer       — per-feature contribution → natural-language text

Hackathon constraints addressed:
  ✓ Explainability (XAI)  — SHAP values, no black box
  ✓ Expert-in-the-loop    — model returns recommendation, not a binding decision
  ✓ Real CSV processing   — pandas pipeline with feature engineering
  ✓ Interpretable output  — human-readable explanation string
"""

import os
import pickle
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

# Columns expected in the hackathon CSV (or produced by the mock builder below).
# Adjust to match the actual contest dataset.
FEATURE_COLS = [
    "revenue_k_tenge",       # Annual revenue, thousands KZT
    "hectares",              # Cultivated area, ha
    "years_operating",       # Years in agribusiness
    "credit_score",          # Internal credit rating 0-100
    "num_employees",         # Full-time workers
    "equipment_value_m",     # Machinery book value, millions KZT
    "num_crops",             # Distinct crop types grown
    "num_livestock_types",   # Livestock categories (0 if none)
    "has_irrigation",        # 1/0
    "owns_land",             # 1/0
    "certified_seeds",       # 1/0
    "tax_status_enc",        # 0=compliant, 1=minor, 2=major  (encoded)
    "subsidy_history_enc",   # 0=none, 1=once, 2=regular, 3=violations
    "loan_history_enc",      # 0=none/good, 1=delayed, 2=bad
    "num_documents",         # Count of submitted supporting documents
    "oblast_priority",       # 1 if oblast is flagged priority region, else 0
]

TARGET_COL = "approved"  # 1 = historically approved/positive, 0 = rejected


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MOCK DATASET  (replace with pd.read_csv("hackathon_data.csv"))
# ──────────────────────────────────────────────────────────────────────────────

def build_mock_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic but realistic dataset of KZ farm subsidy applications.
    Each row represents one application.  Approved ~ 55 % of cases.
    """
    rng = np.random.default_rng(seed)

    revenue        = rng.exponential(scale=40_000, size=n).clip(500, 500_000)
    hectares       = rng.exponential(scale=200, size=n).clip(5, 5_000)
    years_op       = rng.integers(0, 25, size=n)
    credit_score   = rng.integers(20, 100, size=n)
    employees      = rng.integers(1, 100, size=n)
    equipment      = rng.exponential(scale=60, size=n).clip(0, 500)
    num_crops      = rng.integers(0, 6, size=n)
    num_livestock  = rng.integers(0, 4, size=n)
    has_irrig      = rng.integers(0, 2, size=n)
    owns_land      = rng.integers(0, 2, size=n)
    cert_seeds     = rng.integers(0, 2, size=n)
    tax_status     = rng.choice([0, 0, 0, 1, 2], size=n)        # skewed to compliant
    sub_history    = rng.choice([0, 1, 2, 3], p=[.4,.2,.3,.1], size=n)
    loan_hist      = rng.choice([0, 0, 1, 2], size=n)
    num_docs       = rng.integers(0, 7, size=n)
    oblast_prio    = rng.integers(0, 2, size=n)

    # Soft label: approval probability driven by features
    prob = (
        0.20 * (np.log1p(revenue) / np.log1p(500_000))
        + 0.15 * (np.log1p(hectares) / np.log1p(5_000))
        + 0.12 * (years_op / 25)
        + 0.10 * (credit_score / 100)
        + 0.08 * (num_docs / 6)
        - 0.15 * (tax_status / 2)
        - 0.10 * (loan_hist / 2)
        - 0.12 * (sub_history == 3).astype(float)
        + 0.08 * owns_land
        + 0.05 * cert_seeds
        + 0.05 * has_irrig
    ).clip(0.05, 0.95)
    approved = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "revenue_k_tenge":      revenue,
        "hectares":             hectares,
        "years_operating":      years_op,
        "credit_score":         credit_score,
        "num_employees":        employees,
        "equipment_value_m":    equipment,
        "num_crops":            num_crops,
        "num_livestock_types":  num_livestock,
        "has_irrigation":       has_irrig,
        "owns_land":            owns_land,
        "certified_seeds":      cert_seeds,
        "tax_status_enc":       tax_status,
        "subsidy_history_enc":  sub_history,
        "loan_history_enc":     loan_hist,
        "num_documents":        num_docs,
        "oblast_priority":      oblast_prio,
        "approved":             approved,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 3.  RULE ENGINE  (deterministic, always-on fallback + interpretable baseline)
# ──────────────────────────────────────────────────────────────────────────────

def rule_score(row: dict) -> dict:
    """
    Five-block deterministic scoring (max 100 points).
    Mirrors the JS ScoringEngine so frontend and backend are consistent.

    Returns:
        {"total": int, "blocks": {name: (raw, max)}}
    """
    blocks = {}

    # ── Block 1: Financial (0–25) ──────────────────────────────────────────
    f = 0
    rev = row.get("revenue_k_tenge", 0)
    if   rev >= 100_000: f += 8
    elif rev >= 50_000:  f += 6
    elif rev >= 20_000:  f += 4
    elif rev >= 5_000:   f += 2
    f += round((row.get("credit_score", 0) / 100) * 7)
    loan_map = {0: 5, 1: 2, 2: 0}   # 0=none/good, 1=delayed, 2=bad
    f += loan_map.get(int(row.get("loan_history_enc", 0)), 3)
    tax_map  = {0: 5, 1: 2, 2: 0}
    f += tax_map.get(int(row.get("tax_status_enc", 0)), 2)
    blocks["financial"] = (min(25, f), 25)

    # ── Block 2: Production (0–20) ─────────────────────────────────────────
    p = 0
    ha = row.get("hectares", 0)
    if   ha >= 1000: p += 5
    elif ha >= 300:  p += 4
    elif ha >= 100:  p += 3
    elif ha >= 30:   p += 2
    else:            p += 1
    p += min(4, int(row.get("num_crops", 0)))
    p += 2 * int(row.get("has_irrigation", 0))
    p += 2 * int(row.get("owns_land", 0))
    p += 2 * int(row.get("certified_seeds", 0))
    eq = row.get("equipment_value_m", 0)
    if   eq >= 100: p += 3
    elif eq >= 50:  p += 2
    elif eq >= 10:  p += 1
    p += min(2, int(row.get("num_livestock_types", 0)))
    blocks["production"] = (min(20, p), 20)

    # ── Block 3: Reliability (0–20) ────────────────────────────────────────
    r = 0
    yr = row.get("years_operating", 0)
    if   yr >= 10: r += 8
    elif yr >= 5:  r += 6
    elif yr >= 3:  r += 4
    elif yr >= 1:  r += 2
    sub_map = {0: 4, 1: 6, 2: 7, 3: 0}
    r += sub_map.get(int(row.get("subsidy_history_enc", 0)), 3)
    r += 5   # base for being a registered entity
    blocks["reliability"] = (min(20, r), 20)

    # ── Block 4: Compliance / Documents (0–20) ────────────────────────────
    c = 0
    n_docs = int(row.get("num_documents", 0))
    doc_pts = [4, 3, 3, 3, 2, 2]   # value of each successive document
    for i in range(min(n_docs, len(doc_pts))):
        c += doc_pts[i]
    tax = int(row.get("tax_status_enc", 0))
    c += [4, 2, 0][tax]
    if row.get("subsidy_history_enc", 0) != 3:
        c += 2
    blocks["compliance"] = (min(20, c), 20)

    # ── Block 5: Social-economic (0–15) ───────────────────────────────────
    s = 0
    emp = row.get("num_employees", 0)
    if   emp >= 50: s += 5
    elif emp >= 20: s += 4
    elif emp >= 10: s += 3
    elif emp >= 3:  s += 2
    else:           s += 1
    s += min(5, int(row.get("num_crops", 0)) + int(row.get("num_livestock_types", 0)))
    s += 5 * int(row.get("oblast_priority", 0))
    blocks["social"] = (min(15, s), 15)

    total = sum(v for v, _ in blocks.values())
    return {"total": total, "blocks": blocks}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ML MODEL  (XGBoost + SHAP)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "model.pkl"
EXPLAINER_PATH = "shap_explainer.pkl"


def train(df: Optional[pd.DataFrame] = None, save: bool = True) -> dict:
    """
    Train XGBoost on the provided DataFrame (or mock data if None).
    Saves model and SHAP TreeExplainer to disk.

    Returns:
        {"accuracy": float, "roc_auc": float}
    """
    if df is None:
        print("[ml_engine] No DataFrame supplied — using mock dataset (n=200).")
        df = build_mock_dataset()

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"[ml_engine] Training complete | Accuracy={metrics['accuracy']}  AUC={metrics['roc_auc']}")

    explainer = shap.TreeExplainer(model)

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(EXPLAINER_PATH, "wb") as f:
            pickle.dump(explainer, f)
        print(f"[ml_engine] Saved → {MODEL_PATH}, {EXPLAINER_PATH}")

    return metrics


def _load_artefacts():
    """Load persisted model and explainer; raise if not found."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train() first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)
    return model, explainer


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SHAP EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────────

# Human-readable labels for each feature column
FEATURE_LABELS = {
    "revenue_k_tenge":      "годовая выручка",
    "hectares":             "площадь (га)",
    "years_operating":      "лет в агробизнесе",
    "credit_score":         "кредитный рейтинг",
    "num_employees":        "число работников",
    "equipment_value_m":    "стоимость техники",
    "num_crops":            "кол-во культур",
    "num_livestock_types":  "виды животноводства",
    "has_irrigation":       "наличие орошения",
    "owns_land":            "земля в собственности",
    "certified_seeds":      "сертифицированные семена",
    "tax_status_enc":       "налоговый статус",
    "subsidy_history_enc":  "история субсидий",
    "loan_history_enc":     "кредитная история",
    "num_documents":        "кол-во документов",
    "oblast_priority":      "приоритетный регион",
}


def _shap_to_factors(shap_values: np.ndarray, feature_names: list) -> list[dict]:
    """
    Convert raw SHAP values for one sample into a sorted list of factor dicts.
    Positive SHAP → increases approval probability (green).
    Negative SHAP → decreases it (red / yellow).
    """
    factors = []
    for col, sv in zip(feature_names, shap_values):
        label = FEATURE_LABELS.get(col, col)
        factors.append({
            "feature":    col,
            "name":       label,
            "shap_value": round(float(sv), 4),
            "impact":     round(float(sv) * 100, 1),   # scaled to –100…+100 range for UI
        })
    # Most impactful first (by absolute value)
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return factors[:8]   # Top-8 for UI card


def _build_explanation(applicant_name: str, hybrid_score: int,
                       rule_result: dict, factors: list[dict]) -> str:
    """
    Produce a human-readable, non-technical explanation in Russian.
    This satisfies the XAI / Explainability requirement.
    """
    verdict = (
        "соответствует критериям одобрения" if hybrid_score >= 65
        else "находится в зоне условного одобрения" if hybrid_score >= 40
        else "не соответствует базовым критериям субсидирования"
    )

    positive = [f["name"] for f in factors if f["shap_value"] > 0.02][:3]
    negative = [f["name"] for f in factors if f["shap_value"] < -0.02][:3]

    lines = [
        f"{applicant_name} {verdict} — итоговый гибридный балл: {hybrid_score}/100.",
    ]
    if positive:
        lines.append(
            "Факторы, повышающие рейтинг: " + ", ".join(positive) + "."
        )
    if negative:
        lines.append(
            "Факторы, снижающие рейтинг: " + ", ".join(negative) + "."
        )

    rb = rule_result["blocks"]
    weak_blocks = [
        name for name, (raw, mx) in rb.items() if raw / mx < 0.5
    ]
    if weak_blocks:
        block_map = {
            "financial":   "финансовые показатели",
            "production":  "производственный потенциал",
            "reliability": "надёжность и история",
            "compliance":  "документы и соответствие",
            "social":      "социально-экономический эффект",
        }
        labels = [block_map.get(b, b) for b in weak_blocks]
        lines.append(
            "Блоки с низким баллом: " + ", ".join(labels) + " — рекомендуется запросить уточнения."
        )

    if hybrid_score >= 65:
        lines.append(
            "Рекомендация модели: одобрить заявку. "
            "Эксперту следует верифицировать ключевые документы перед финализацией."
        )
    elif hybrid_score >= 40:
        lines.append(
            "Рекомендация модели: условное одобрение. "
            "Необходимо устранить выявленные замечания и предоставить подтверждающие документы."
        )
    else:
        lines.append(
            "Рекомендация модели: отклонить текущую заявку. "
            "Рекомендуется повторная подача после устранения блокирующих факторов."
        )

    lines.append(
        "\n⚠ Данный вывод является рекомендацией модели и не заменяет решение уполномоченного эксперта."
    )

    return " ".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN SCORING FUNCTION  (called by FastAPI)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringInput:
    """Mirrors the JSON payload sent from the HTML frontend."""
    # Identification
    iin:  str = ""
    name: str = "Заявитель"
    # Numeric features
    revenue_k_tenge:     float = 0.0
    hectares:            float = 0.0
    years_operating:     int   = 0
    credit_score:        int   = 50
    num_employees:       int   = 1
    equipment_value_m:   float = 0.0
    num_crops:           int   = 0
    num_livestock_types: int   = 0
    # Binary flags
    has_irrigation:   int = 0
    owns_land:        int = 1
    certified_seeds:  int = 0
    # Encoded categoricals
    tax_status_enc:      int = 0   # 0=compliant, 1=minor, 2=major
    subsidy_history_enc: int = 0   # 0=none, 1=once, 2=regular, 3=violations
    loan_history_enc:    int = 0   # 0=none/good, 1=delayed, 2=bad
    num_documents:       int = 0
    oblast_priority:     int = 0


def score(inp: ScoringInput) -> dict:
    """
    Full scoring pipeline: Rule engine → ML prediction → SHAP → Hybrid score.

    Falls back gracefully to rule-only score if model artefacts are missing,
    ensuring the system is always available (important for hackathon demo).
    """
    row = inp.__dict__

    # ── Step 1: Rule engine (always runs) ──────────────────────────────────
    rule_result = rule_score(row)
    rule_total  = rule_result["total"]

    # ── Step 2: ML prediction + SHAP (runs only if model exists) ──────────
    ml_prob     = None
    shap_factors = []
    ml_available = os.path.exists(MODEL_PATH)

    if ml_available:
        try:
            model, explainer = _load_artefacts()
            feature_vector = np.array([[
                row.get(col, 0) for col in FEATURE_COLS
            ]], dtype=float)
            feature_df = pd.DataFrame(feature_vector, columns=FEATURE_COLS)

            ml_prob = float(model.predict_proba(feature_df)[0, 1])
            sv      = explainer.shap_values(feature_df)

            # XGBoost TreeExplainer returns ndarray of shape (1, n_features)
            if isinstance(sv, list):
                sv = sv[1]    # class-1 SHAP values for binary classifiers
            shap_factors = _shap_to_factors(sv[0], FEATURE_COLS)
        except Exception as exc:
            print(f"[ml_engine] ML inference failed: {exc} — falling back to rules only.")
            ml_available = False

    # ── Step 3: Hybrid score ───────────────────────────────────────────────
    if ml_available and ml_prob is not None:
        ml_score_scaled = round(ml_prob * 100)
        hybrid_score    = round(0.4 * rule_total + 0.6 * ml_score_scaled)
        engine_used     = "hybrid (rules 40 % + XGBoost 60 %)"
    else:
        hybrid_score = rule_total
        ml_score_scaled = None
        engine_used  = "rule-only (model not trained yet)"

    hybrid_score = max(0, min(100, hybrid_score))

    # ── Step 4: Explanation text (XAI) ────────────────────────────────────
    # If SHAP is available use it; otherwise fall back to rule-block explanation
    if not shap_factors:
        # Build pseudo-factors from rule blocks for UI consistency
        block_label_map = {
            "financial":   "Финансовая состоятельность",
            "production":  "Производственный потенциал",
            "reliability": "Надёжность и история",
            "compliance":  "Документы и соответствие",
            "social":      "Социально-экономический эффект",
        }
        for bname, (raw, mx) in rule_result["blocks"].items():
            midpoint = mx / 2
            shap_factors.append({
                "feature":    bname,
                "name":       block_label_map.get(bname, bname),
                "shap_value": round((raw - midpoint) / mx, 4),
                "impact":     round((raw - midpoint) / mx * 100, 1),
            })
        shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    explanation = _build_explanation(inp.name, hybrid_score, rule_result, shap_factors)

    return {
        "score":          hybrid_score,
        "rule_score":     rule_total,
        "ml_score":       ml_score_scaled,
        "engine_used":    engine_used,
        "score_blocks":   {
            k: {"raw": v[0], "max": v[1], "pct": round(v[0] / v[1] * 100)}
            for k, v in rule_result["blocks"].items()
        },
        "shap_factors":   shap_factors,
        "explanation":    explanation,
        "verdict":        (
            "approve" if hybrid_score >= 65
            else "review" if hybrid_score >= 40
            else "reject"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI  —  python ml_engine.py
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("АгроСкоринг KZ  |  ML Engine  |  Stage 1 Demo")
    print("=" * 60)

    # 1. Build and display mock dataset
    df = build_mock_dataset(n=300)
    print(f"\nMock dataset: {len(df)} rows, {len(FEATURE_COLS)} features")
    print(f"Approval rate: {df['approved'].mean():.1%}\n")

    # 2. Train model
    metrics = train(df)
    print(f"\nModel metrics: {metrics}")

    # 3. Score a sample applicant
    sample = ScoringInput(
        iin="120540001234",
        name="ИП «Жайлау»",
        revenue_k_tenge=45_000,
        hectares=250,
        years_operating=6,
        credit_score=72,
        num_employees=12,
        equipment_value_m=65,
        num_crops=3,
        num_livestock_types=1,
        has_irrigation=1,
        owns_land=1,
        certified_seeds=1,
        tax_status_enc=0,
        subsidy_history_enc=2,
        loan_history_enc=0,
        num_documents=5,
        oblast_priority=1,
    )

    result = score(sample)

    print("\n" + "─" * 60)
    print(f"Applicant  : {sample.name}  ({sample.iin})")
    print(f"Hybrid score : {result['score']}/100  ({result['verdict'].upper()})")
    print(f"Rule score   : {result['rule_score']}/100")
    print(f"ML score     : {result['ml_score']}")
    print(f"Engine       : {result['engine_used']}")
    print("\nTop SHAP factors:")
    for f in result["shap_factors"][:5]:
        sign = "▲" if f["shap_value"] > 0 else "▼"
        print(f"  {sign} {f['name']:<35}  SHAP={f['shap_value']:+.4f}")
    print("\nExplanation:")
    print(textwrap.fill(result["explanation"], width=70, initial_indent="  ", subsequent_indent="  "))
    print("─" * 60)
