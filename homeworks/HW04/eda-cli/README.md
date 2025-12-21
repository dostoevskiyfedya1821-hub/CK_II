# HW04 – eda_cli: HTTP-сервис качества датасетов

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (HW04):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

Команда генерирует полный EDA-отчёт с таблицами и графиками.

```bash
uv run eda-cli report data/example.csv --out-dir reports
```
Новые параметры команды `report`

| Опция | Описание | Тип   | По умолчанию                                          |
|-------|-----------|-------|-------------------------------------------------------|
| `--title` | Заголовок, который будет использоваться в отчете report.md. | STR   | `"Data Quality and Exploratory Data Analysis Report"` |
| `--min-missing-share` | Порог доли пропусков (от 0.0 до 1.0), выше которого колонка считается проблемной и попадает в отдельный список в отчете. | FLOAT | `0.01` (1%)                                           |
| `--max-hist-columns` | Максимум числовых колонок для включения в набор гистограмм. | INT   | `6`                                                   |
| `--json-summary` | Сохранить компактную сводку датасета (`n_rows`, `quality_score`, проблемные колонки) в файл `summary.json` (Доп. часть). | BOOL  | `False`                                               |

#### Пример вызова с новыми опциями

```bash
uv run eda-cli report data/example.csv --out-dir reports_hw04 --title "Финальный отчет HW04" --min-missing-share 0.1 --json-summary
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.json` – компактная JSON-сводка (при использовании `--json-summary`);
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```

## Дополнительная часть (Вариант B)

### JSON-сводка по датасету

В команду `report` добавлена опция `--json-summary`.

При ее использовании генерируется файл `summary.json`, который содержит:

1. **Размеры датасета** (`n_rows`, `n_cols`)
2. **Интегральный показатель качества** (`quality_score`)
3. **Перечень флагов качества** (например, `has_constant_columns`)
4. **Список проблемных колонок**, выявленных эвристиками (например, с долей пропусков более 50% или константных)

## HTTP API (HW04)

Запуск сервиса:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Эндпоинты:

- GET /health

- POST /quality

- POST /quality-from-csv

- POST /quality-flags-from-csv (доп. HW04)

Пример:

```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/quality-from-csv" -F "file=@data/example.csv"
curl -X POST "http://127.0.0.1:8000/quality-flags-from-csv" -F "file=@data/example.csv"
```