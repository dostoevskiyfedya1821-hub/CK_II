from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import summarize_dataset, missing_table, compute_quality_flags


app = FastAPI(
    title="EDA CLI Dataset Quality API",
    version="0.1.0",
    description="HTTP-сервис качества датасетов поверх eda_cli (HW04).",
    docs_url="/docs",
    redoc_url=None,
)


# -------------------- Pydantic модели --------------------

class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    categorical_cols: int = Field(..., ge=0)


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    message: str
    latency_ms: float = Field(..., ge=0.0)
    flags: Dict[str, Any]
    dataset_shape: Optional[Dict[str, int]] = None


class QualityFlagsResponse(BaseModel):
    flags: Dict[str, Any]


# -------------------- Endpoints --------------------

@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "eda-cli-api", "version": app.version}


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Заглушка: оценивает качество по агрегированным признакам (как в семинаре).
    """
    start = perf_counter()

    score = 1.0
    score -= float(req.max_missing_share)

    # простые эвристики (мягкие)
    if req.n_rows < 100:
        score -= 0.2
    if req.n_cols > 100:
        score -= 0.1
    if req.numeric_cols == 0:
        score -= 0.05
    if req.categorical_cols == 0:
        score -= 0.03

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    flags = {
        "too_few_rows": req.n_rows < 100,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_cols": req.numeric_cols == 0,
        "no_categorical_cols": req.categorical_cols == 0,
    }

    latency_ms = (perf_counter() - start) * 1000.0
    message = "OK for model" if ok_for_model else "Not OK for model"

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


@app.post("/quality-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Принимает CSV, считает quality_score и flags через твой EDA-core (HW03).
    Важно: возвращаем flags ЦЕЛИКОМ, чтобы были видны твои новые метрики.
    """
    start = perf_counter()

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV пустой (получился пустой DataFrame).")

    summary = summarize_dataset(df)
    miss = missing_table(df)

    # Вызываем так, чтобы 100% отработали твои новые эвристики HW03 (нули/unique)
    flags_all = compute_quality_flags(df, summary, miss)
    flags_bool = {k: bool(v) for k, v in flags_all.items() if isinstance(v, bool)}

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    latency_ms = (perf_counter() - start) * 1000.0
    message = "OK for model" if ok_for_model else "Not OK for model"

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": int(summary.n_rows), "n_cols": int(summary.n_cols)},
    )


# --- ОБЯЗАТЕЛЬНЫЙ НОВЫЙ ЭНДПОИНТ HW04 (вариант A) ---

@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse, tags=["quality"])
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Принимает CSV, считает quality_score через EDA-core (HW03).
    Для /quality-from-csv: “flags — только bool для совместимости”
    Для /quality-flags-from-csv: “flags — полный словарь (включая списки/числа)”
    """

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV пустой (получился пустой DataFrame).")

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags_all = compute_quality_flags(df, summary, miss)

    return QualityFlagsResponse(flags=flags_all)
