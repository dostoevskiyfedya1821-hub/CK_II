from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes

import json
from pathlib import Path


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result

def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    *,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Считает quality-флаги и интегральный quality_score.
    df опционален: нужен для эвристик по нулям/unique (если не передан — считаем по summary).
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # --- Константные колонки / min_unique_values_count ---
    if df is not None and not df.empty:
        unique_non_null = df.nunique(dropna=True)
        non_null_counts = df.notna().sum()

        constant_cols = [
            col for col in df.columns
            if non_null_counts[col] > 0 and unique_non_null[col] <= 1
        ]
        min_unique = int(unique_non_null.min()) if len(unique_non_null) else 0
    else:
        constant_cols = [c.name for c in summary.columns if c.non_null > 0 and c.unique <= 1]
        min_unique = min([c.unique for c in summary.columns if c.non_null > 0], default=0)

    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns"] = constant_cols
    flags["min_unique_values_count"] = int(min_unique)

    # --- Нули в числовых / max_zero_values_share ---
    ZERO_SHARE_THRESHOLD = 0.8
    flags["zero_share_threshold"] = ZERO_SHARE_THRESHOLD

    if df is not None and not df.empty and summary.n_rows > 0:
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            zero_share = (numeric_df == 0).sum() / summary.n_rows
            max_zero_share = float(zero_share.max())
        else:
            max_zero_share = 0.0
    else:
        max_zero_share = 0.0

    flags["max_zero_values_share"] = float(max_zero_share)
    flags["has_many_zero_values"] = max_zero_share > ZERO_SHARE_THRESHOLD

    # --- quality_score ---
    score = 1.0
    score -= max_missing_share

    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["min_unique_values_count"] <= 1:
        score -= 0.15
    if flags["max_zero_values_share"] > ZERO_SHARE_THRESHOLD:
        score -= 0.10

    flags["quality_score"] = max(0.0, min(1.0, score))
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)


# ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ВАРИАНТА B (JSON-сводка)

def numpy_json_encoder(obj):
    """Универсальный кодировщик для преобразования типов NumPy/Pandas в нативные Python-типы."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        # Если в JSON-структуру попал DataFrame или Series,
        # преобразуем его в словарь/список, чтобы избежать ошибки
        return obj.to_dict() if isinstance(obj, pd.Series) else obj.to_dict('records')

    # Если тип не распознан, вызываем стандартную ошибку сериализации
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# src/eda_cli/core.py (обновление функции make_json_summary)

def make_json_summary(
    df: pd.DataFrame,
    summary: DatasetSummary,
    quality_flags: Dict[str, Any],
    missing_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Формирует словарь с компактной сводкой данных для сохранения в JSON.
    Включает размеры, quality_score и список проблемных колонок.
    """

    # 1. Сбор базовых метрик и качества
    json_data = {
        "dataset_size": {
            # Явное приведение в int для n_rows/n_cols на всякий случай
            "n_rows": int(summary.n_rows),
            "n_cols": int(summary.n_cols),
        },
        # Обязательное приведение quality_score к нативному float
        "quality_score": float(quality_flags["quality_score"]),
        "quality_flags": {k: v for k, v in quality_flags.items() if k not in ["quality_score"]},
    }

    # 2. Список "проблемных" колонок
    problem_cols: List[Dict[str, Any]] = []

    # Порог для определения проблемных пропусков
    HIGH_MISSING_THRESHOLD = 0.5

    # Эвристика 1: колонки с очень большим количеством пропусков (> 50%)
    high_missing_cols = missing_df[missing_df["missing_share"] > HIGH_MISSING_THRESHOLD]
    for col_name in high_missing_cols.index:
        problem_cols.append({
            "column": col_name,
            "reason": "high_missing_share",
            # !!! ЯВНОЕ ПРИВЕДЕНИЕ К float !!!
            "value": float(missing_df.loc[col_name, "missing_share"]),
            "threshold": HIGH_MISSING_THRESHOLD,
        })

    # Эвристика 2: константные колонки (уникальных <= 1)
    if quality_flags["has_constant_columns"]:
        unique_non_null = df.nunique(dropna=True)
        non_null_counts = df.notna().sum()

        constant_cols_names = [
            col for col in df.columns
            if non_null_counts[col] > 0 and unique_non_null[col] <= 1
        ]

        for col_name in constant_cols_names:
            if col_name not in [pc["column"] for pc in problem_cols]:
                problem_cols.append({
                    "column": col_name,
                    "reason": "is_constant",
                    "value": int(unique_non_null[col_name]),
                    "threshold": 1,
                })

    json_data["problem_columns"] = problem_cols

    return json_data


def save_json_summary(data: Dict[str, Any], path: Path) -> None:
    """Сохраняет словарь в JSON-файл."""

    with path.open("w", encoding="utf-8") as f:
        # !!! КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: default=numpy_json_encoder !!!
        json.dump(data, f, ensure_ascii=False, indent=4, default=numpy_json_encoder)