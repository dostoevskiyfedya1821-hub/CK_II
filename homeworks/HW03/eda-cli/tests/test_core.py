from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    DatasetSummary,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": pd.Series([10, 20, 30, None], dtype="float"),
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    # ИСПРАВЛЕНИЕ: Передаем df как первый аргумент
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert not corr.empty
    assert "age" in corr.columns

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# НОВЫЙ ТЕСТ ДЛЯ ПРОВЕРКИ НОВЫХ ЭВРИСТИК (ЗАДАНИЕ 2.3.3)

def test_has_constant_columns_flag():
    """
    Проверяет, что флаг 'has_constant_columns' корректно выставляется,
    а quality_score снижается.
    """
    # Создание DataFrame с константной колонкой
    df = pd.DataFrame(
        {
            "id_col": [101, 102, 103, 104],
            "const_col": [500, 500, 500, 500],  # Константная колонка
            "nan_col": [np.nan, np.nan, np.nan, np.nan],  # Колонка с 100% пропусков
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # Вызов функции
    flags = compute_quality_flags(summary, missing_df, df=df)

    # 1. Проверка флага константности
    # ИСПРАВЛЕНО: используем '==' вместо 'is'
    assert flags["has_constant_columns"] == True, "Флаг has_constant_columns должен быть True."

    # 2. Проверка флага нулей
    assert flags["has_many_zero_values"] == False, "Флаг has_many_zero_values должен быть False."

    # 3. Проверка корректировки Quality Score
    # Расчет: 1.0 (начальный) - 1.0 (за пропуски) - 0.15 (за константу) = -0.15 -> 0.0
    expected_score = 0.0

    assert abs(flags["quality_score"] - expected_score) < 0.01, "Quality score не был скорректирован корректно."

    # Дополнительная проверка (если нужна): убедимся, что min_unique_values_count < 2
    assert flags["min_unique_values_count"] <= 1, "Минимальное число уникальных значений должно быть 0 или 1."