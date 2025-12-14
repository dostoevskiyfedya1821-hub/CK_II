from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    # --- ВАРИАНТ B: ИМПОРТ НОВЫХ ФУНКЦИЙ ---
    make_json_summary,
    save_json_summary,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
        path: Path,
        sep: str = ",",
        encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),

        # Существующий параметр
        max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),

        # Новые параметры
        title: str = typer.Option(
            "Data Quality and Exploratory Data Analysis Report",
            help="Заголовок, который будет использоваться в report.md.",
        ),
        min_missing_share: float = typer.Option(
            0.01,
            min=0.0,
            max=1.0,
            help="Доля пропусков (от 0.0 до 1.0), выше которой колонка считается проблемной.",
        ),
        # --- ВАРИАНТ B: НОВЫЙ ПАРАМЕТР ---
        json_summary: bool = typer.Option(
            False,
            "--json-summary",
            help="Сохранить компактную сводку датасета в файл summary.json."
        ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Вычисления
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(df, summary, missing_df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # --- ВАРИАНТ B: ЛОГИКА СОХРАНЕНИЯ JSON ---
    json_path = out_root / "summary.json"
    if json_summary:
        # Убедитесь, что core.make_json_summary принимает df как первый аргумент
        json_data = make_json_summary(df, summary, quality_flags, missing_df)
        save_json_summary(json_data, json_path)
    # ---------------------------------------------

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        # Используем новый параметр --title
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        # --- ВАРИАНТ B: УПОМИНАНИЕ JSON В MD ---
        if json_summary:
            f.write("## Компактная сводка\n\n")
            f.write(
                f"Полные данные о размерах, Quality Score и проблемных колонках доступны в файле `summary.json`.\n\n")
        # ---------------------------------------------

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Интегральный показатель (Quality Score): **{quality_flags['quality_score']:.3f}**\n")

        # Включаем новую настройку порога из CLI
        f.write(f"- Порог для проблемных пропусков (--min-missing-share): **{min_missing_share:.2%}**\n")

        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")

        # Добавляем новые флаги из core.py
        f.write(f"- Есть константные колонки: **{quality_flags['has_constant_columns']}**\n")
        f.write(
            f"- Есть колонки со слишком большим количеством нулей (>{quality_flags['zero_share_threshold']:.0%}): **{quality_flags['has_many_zero_values']}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Проблемные колонки (Missing Data)\n\n")

        # Используем новый параметр --min-missing-share для фильтрации
        problem_cols = missing_df[missing_df["missing_share"] >= min_missing_share]

        if problem_cols.empty:
            f.write("Нет колонок, где доля пропусков выше заданного порога.\n\n")
        else:
            f.write(f"Следующие колонки имеют долю пропусков ≥ **{min_missing_share:.2%}**:\n\n")
            f.write(problem_cols.to_markdown(floatfmt=".2%"))
            f.write("\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")

    # --- ВАРИАНТ B: ВЫВОД ПУТИ К JSON ---
    if json_summary:
        typer.echo(f"- JSON-сводка: {json_path}")
    # ------------------------------------

    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()