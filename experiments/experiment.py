"""Run paper experiments against the already-running backend."""

from __future__ import annotations

import base64
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
import pandas as pd
import typer
from tqdm import tqdm

try:
    from experiments.test_cases import (
        CHART_INTERP_CASES,
        DISPLAY_CARD_CASES,
        KG_ABLATION_CHART_CASES,
        NL_SQL_CASES,
        PLOT_CASES,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from test_cases import CHART_INTERP_CASES, DISPLAY_CARD_CASES, KG_ABLATION_CHART_CASES, NL_SQL_CASES, PLOT_CASES


app = typer.Typer(add_completion=False, help="Run the paper experiments against the backend.")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results"
FLOAT_TOLERANCE = 0.01
TRUTHY_VALUES = {"1", "true", "yes", "on"}
TASK_NAMES = {"nl_sql", "display_cards", "plot_gen", "chart_interp", "kg_ablation"}
KG_RANGES = {
    "temperature_c": (20.0, 26.0),
    "humidity_rh": (30.0, 60.0),
}
MODEL_MENU = [
    ("Ministral 3 3B", "ministral-3:3b"),
    ("Qwen 3 VL 4B", "qwen3-vl:4b"),
    ("Gemma 3 4B IT QAT", "gemma3:4b-it-qat"),
    ("Granite 3.2 Vision", "granite3.2-vision:latest"),
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "default"


def menu_model_names() -> List[str]:
    return [model for _, model in MODEL_MENU]


def parse_model_selection(raw: Optional[str], menu: List[str]) -> List[str]:
    if not raw:
        return []
    value = raw.strip()
    if value.lower() in {"all", "*"}:
        return list(menu)
    parts = [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]
    models: List[str] = []
    seen = set()
    for part in parts:
        if part.isdigit():
            index = int(part)
            if index < 1 or index > len(menu):
                raise ValueError(f"Model number {index} is outside 1-{len(menu)}.")
            model = menu[index - 1]
        else:
            model = part
        if model not in seen:
            models.append(model)
            seen.add(model)
    return models


def prompt_model_selection() -> List[str]:
    menu = menu_model_names()
    typer.echo("Select model(s) to evaluate:")
    for index, (label, model) in enumerate(MODEL_MENU, start=1):
        typer.echo(f"  {index}. {label} ({model})")
    typer.echo("Enter one number, comma-separated numbers, or 'all'.")
    while True:
        selection = typer.prompt("Model selection", default="1")
        try:
            models = parse_model_selection(selection, menu)
        except ValueError as exc:
            typer.echo(f"Invalid model selection: {exc}", err=True)
            continue
        if models:
            return models
        typer.echo("Select at least one model.", err=True)


def parse_models(models_arg: Optional[str], client: httpx.Client, backend_url: str) -> List[str]:
    menu = menu_model_names()
    if models_arg:
        try:
            models = parse_model_selection(models_arg, menu)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if not models:
            raise typer.BadParameter("--models was provided but no model names were parsed.")
        return models
    if sys.stdin.isatty():
        return prompt_model_selection()
    try:
        response = client.get(f"{backend_url}/health")
        response.raise_for_status()
        payload = response.json()
        if payload.get("model"):
            return [str(payload["model"])]
    except Exception:
        pass
    return ["default"]


def load_merged_dataframe() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(DATA_DIR.glob("*_measurements_IAQ_1h_avg.csv")):
        frame = pd.read_csv(csv_path)
        frame["Date"] = pd.to_datetime(frame["Date"])
        if "location" not in frame.columns:
            frame["location"] = csv_path.name.split("_")[0]
        frame["location"] = frame["location"].astype(str).str.lower()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No IAQ CSV files found under {DATA_DIR}")
    dataframe = pd.concat(frames, ignore_index=True)
    dataframe["date_only"] = dataframe["Date"].dt.strftime("%Y-%m-%d")
    return dataframe


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_static_path(path: Path) -> str:
    experiments_root = REPO_ROOT / "experiments"
    try:
        relative = path.resolve().relative_to(experiments_root.resolve())
        return f"/static/{relative.as_posix()}"
    except ValueError:
        return path.resolve().as_posix()


def decode_png(image_b64: str) -> bytes:
    data = base64.b64decode(image_b64, validate=True)
    if not data or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Response did not contain a valid PNG image.")
    return data


def is_close(left: Optional[float], right: Optional[float], tolerance: float = FLOAT_TOLERANCE) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def extract_scalar(rows: Any) -> Optional[float]:
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    if not isinstance(rows[0], list) or len(rows[0]) != 1:
        return None
    try:
        return float(rows[0][0])
    except (TypeError, ValueError):
        return None


def classify_display(metric: str, value: float, kg_disabled: bool = False) -> str:
    if kg_disabled:
        return "unknown"
    bounds = KG_RANGES.get(metric)
    if not bounds:
        return "unknown"
    minimum, maximum = bounds
    if value < minimum:
        return "warn-low"
    if value > maximum:
        return "warn-high"
    return "ok"


def record_latency(record: Dict[str, Any]) -> Optional[float]:
    for key in ("latency_total_s", "latency_s", "latency_interp_s", "latency_plot_s"):
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def print_summary_table(records: List[Dict[str, Any]]) -> None:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        model = str(record.get("model") or "default")
        task = str(record.get("task") or "unknown")
        if record.get("condition"):
            task = f"{task} [{record['condition']}]"
        bucket = grouped.setdefault(
            (model, task),
            {"n": 0, "correct": 0, "pending": 0, "latencies": []},
        )
        bucket["n"] += 1
        if record.get("correct") is True:
            bucket["correct"] += 1
        elif record.get("correct") is None:
            bucket["pending"] += 1
        latency = record_latency(record)
        if latency is not None:
            bucket["latencies"].append(latency)

    headers = ["Model", "Task", "N", "Correct", "Pending", "Accuracy", "Mean Lat (s)"]
    rows: List[List[str]] = []
    for (model, task), bucket in sorted(grouped.items()):
        graded = bucket["n"] - bucket["pending"]
        accuracy = "-" if graded == 0 else f"{(bucket['correct'] / graded) * 100:.1f}%"
        mean_latency = "-" if not bucket["latencies"] else f"{sum(bucket['latencies']) / len(bucket['latencies']):.3f}"
        rows.append(
            [
                model,
                task,
                str(bucket["n"]),
                str(bucket["correct"]),
                str(bucket["pending"]),
                accuracy,
                mean_latency,
            ]
        )

    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    typer.echo(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    typer.echo("-+-".join("-" * width for width in widths))
    for row in rows:
        typer.echo(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


@dataclass
class ExperimentContext:
    backend_url: str
    output_dir: Path
    client: httpx.Client
    dataframe: pd.DataFrame
    run_timestamp: str


class DataOracle:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def nl_sql_expected(self, case: Dict[str, Any]) -> float:
        subset = self.dataframe[self.dataframe["location"].str.lower() == case["location"].lower()]
        series = subset[case["expected_col"]].dropna()
        agg_name = case["expected_agg"]
        if agg_name == "mean":
            return float(series.mean())
        if agg_name == "min":
            return float(series.min())
        if agg_name == "max":
            return float(series.max())
        raise ValueError(f"Unsupported aggregate: {agg_name}")

    def display_expected(self, case: Dict[str, Any], kg_disabled: bool = False) -> Dict[str, Any]:
        subset = self.dataframe[
            (self.dataframe["date_only"] == case["date"])
            & (self.dataframe["location"].str.lower() == case["location"].lower())
        ]
        if subset.empty:
            raise ValueError(f"No data for {case['date']} / {case['location']}")
        value = float(subset[case["metric"]].mean())
        return {
            "value": value,
            "state": classify_display(case["metric"], value, kg_disabled=kg_disabled),
        }

    def safe_limit_expected(self, case: Dict[str, Any]) -> Dict[str, Any]:
        metric = str(case["expected_metric"])
        bounds = KG_RANGES.get(metric)
        if bounds is None:
            raise ValueError(f"No KG safe range configured for {metric}")
        minimum, maximum = bounds
        subset = self.dataframe[
            (self.dataframe["date_only"] == case["expected_date"])
            & (self.dataframe["location"].str.lower() == case["expected_location"].lower())
        ]
        series = subset[metric].dropna()
        if series.empty:
            raise ValueError(f"No data for {case['expected_date']} / {case['expected_location']} / {metric}")
        observed_min = float(series.min())
        observed_max = float(series.max())
        within = observed_min >= minimum and observed_max <= maximum
        if within:
            violation = None
        elif observed_min < minimum and observed_max > maximum:
            violation = "below_and_above"
        elif observed_min < minimum:
            violation = "below"
        else:
            violation = "above"
        return {
            "expected_safe_state": "within_safe_limit" if within else "outside_safe_limit",
            "recommended_min": minimum,
            "recommended_max": maximum,
            "observed_min": round(observed_min, 6),
            "observed_max": round(observed_max, 6),
            "violation": violation,
        }


class BaseExperiment:
    task_name = "base"

    def __init__(self, context: ExperimentContext, oracle: DataOracle):
        self.context = context
        self.oracle = oracle

    def result_path(self, model: str) -> Path:
        return self.context.output_dir / f"{self.task_name}_{sanitize_name(model)}_{self.context.run_timestamp}.jsonl"

    def run_model(self, model: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NLToSQLExperiment(BaseExperiment):
    task_name = "nl_sql"

    def run_model(self, model: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for case in tqdm(NL_SQL_CASES, desc=f"{self.task_name}:{model}", unit="case"):
            expected_value = self.oracle.nl_sql_expected(case)
            record: Dict[str, Any] = {
                "task": self.task_name,
                "question": case["question"],
                "generated_sql": None,
                "result_value": None,
                "expected_value": round(expected_value, 6),
                "correct": False,
                "latency_s": None,
                "model": model,
                "timestamp": utc_now_iso(),
            }
            started = time.perf_counter()
            try:
                response = self.context.client.post(
                    f"{self.context.backend_url}/api/data/ask",
                    json={"question": case["question"], "model": model, "kg_disabled": False},
                )
                record["latency_s"] = round(time.perf_counter() - started, 4)
                if response.status_code != 200:
                    record["error"] = response.text
                    records.append(record)
                    continue
                payload = response.json()
                generated_sql = str(payload.get("sql") or "")
                result_value = extract_scalar(payload.get("rows"))
                valid_sql = generated_sql.strip().lower().startswith(("select", "with"))
                record["generated_sql"] = generated_sql
                record["result_value"] = result_value
                record["correct"] = bool(valid_sql and is_close(result_value, expected_value))
            except Exception as exc:
                record["latency_s"] = round(time.perf_counter() - started, 4)
                record["error"] = str(exc)
            records.append(record)
        return records


class DisplayCardsExperiment(BaseExperiment):
    task_name = "display_cards"

    def evaluate_case(
        self,
        case: Dict[str, Any],
        model: str,
        kg_disabled: bool = False,
        condition: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_value = task_name or self.task_name
        expected = self.oracle.display_expected(case, kg_disabled=kg_disabled)
        record: Dict[str, Any] = {
            "task": task_value,
            "query_params": {
                "date": case["date"],
                "location": case["location"],
                "time": case["time"],
                "metric": case["metric"],
            },
            "returned_value": None,
            "returned_state": None,
            "expected_value": round(expected["value"], 6),
            "expected_state": expected["state"],
            "correct": False,
            "latency_s": None,
            "model": model,
            "timestamp": utc_now_iso(),
        }
        if condition:
            record["condition"] = condition

        params = {"date": case["date"], "location": case["location"]}
        if kg_disabled:
            params["kg_disabled"] = "true"
        headers = {"X-KG-Disabled": "true"} if kg_disabled else None

        started = time.perf_counter()
        try:
            response = self.context.client.get(
                f"{self.context.backend_url}/api/display/cards",
                params=params,
                headers=headers,
            )
            record["latency_s"] = round(time.perf_counter() - started, 4)
            if response.status_code != 200:
                record["error"] = response.text
                return record
            payload = response.json()
            cards = payload.get("cards") or []
            card = next((item for item in cards if item.get("title") == case["metric"]), None)
            if not card:
                record["error"] = f"Metric {case['metric']} missing from response."
                return record
            returned_value = float(card["value"])
            returned_state = str(card["status"])
            record["returned_value"] = returned_value
            record["returned_state"] = returned_state
            record["correct"] = bool(
                returned_state == expected["state"] and is_close(returned_value, expected["value"])
            )
        except Exception as exc:
            record["latency_s"] = round(time.perf_counter() - started, 4)
            record["error"] = str(exc)
        return record

    def run_model(self, model: str) -> List[Dict[str, Any]]:
        return [
            self.evaluate_case(case, model=model, kg_disabled=False)
            for case in tqdm(DISPLAY_CARD_CASES, desc=f"{self.task_name}:{model}", unit="case")
        ]


class PlotGenerationExperiment(BaseExperiment):
    task_name = "plot_gen"

    def run_model(self, model: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for case in tqdm(PLOT_CASES, desc=f"{self.task_name}:{model}", unit="case"):
            record: Dict[str, Any] = {
                "task": self.task_name,
                "request_text": case["request"],
                "parsed_metric": None,
                "parsed_location": None,
                "parsed_date": None,
                "correct": False,
                "latency_s": None,
                "model": model,
                "timestamp": utc_now_iso(),
            }
            started = time.perf_counter()
            try:
                response = self.context.client.post(
                    f"{self.context.backend_url}/api/data/plot",
                    json={"question": case["request"], "model": model, "kg_disabled": False},
                )
                record["latency_s"] = round(time.perf_counter() - started, 4)
                if response.status_code != 200:
                    record["error"] = response.text
                    records.append(record)
                    continue
                payload = response.json()
                image_bytes = decode_png(str(payload.get("image_b64") or ""))
                parsed = payload.get("parsed") or {}
                parsed_metric = parsed.get("metric")
                parsed_location = parsed.get("location")
                parsed_date = parsed.get("date_from")
                record["parsed_metric"] = parsed_metric
                record["parsed_location"] = parsed_location
                record["parsed_date"] = parsed_date
                record["correct"] = bool(
                    image_bytes
                    and parsed_metric == case["expected_metric"]
                    and parsed_location == case["expected_location"]
                    and parsed_date == case["expected_date"]
                )
            except Exception as exc:
                record["latency_s"] = round(time.perf_counter() - started, 4)
                record["error"] = str(exc)
            records.append(record)
        return records


class ChartInterpretationExperiment(BaseExperiment):
    task_name = "chart_interp"

    def chart_dir(self, model: str) -> Path:
        return self.context.output_dir / "charts" / sanitize_name(model)

    def response_dir(self, model: str) -> Path:
        return self.context.output_dir / "responses" / sanitize_name(model)

    def _artifact_stem(self, index: int, case: Dict[str, Any], condition: Optional[str]) -> str:
        base = sanitize_name(case["plot_request"])
        suffix = condition or "default"
        return f"{index:02d}_{suffix}_{base}"

    def evaluate_case(
        self,
        case: Dict[str, Any],
        model: str,
        index: int,
        kg_disabled: bool = False,
        condition: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_value = task_name or self.task_name
        record: Dict[str, Any] = {
            "task": task_value,
            "plot_request": case["plot_request"],
            "question": case["question"],
            "model_response": None,
            "chart_path": None,
            "response_path": None,
            "correct": None,
            "latency_plot_s": None,
            "latency_interp_s": None,
            "latency_total_s": None,
            "model": model,
            "timestamp": utc_now_iso(),
        }
        for key in ("rubric", "expected_metric", "expected_location", "expected_date"):
            if key in case:
                record[key] = case[key]
        if condition:
            record["condition"] = condition

        headers = {"X-KG-Disabled": "true"} if kg_disabled else None
        plot_started = time.perf_counter()
        image_bytes: Optional[bytes] = None
        try:
            plot_response = self.context.client.post(
                f"{self.context.backend_url}/api/data/plot",
                json={"question": case["plot_request"], "model": model, "kg_disabled": kg_disabled},
                headers=headers,
            )
            record["latency_plot_s"] = round(time.perf_counter() - plot_started, 4)
            if plot_response.status_code != 200:
                record["error"] = plot_response.text
                return record
            plot_payload = plot_response.json()
            image_bytes = decode_png(str(plot_payload.get("image_b64") or ""))
            artifact_stem = self._artifact_stem(index, case, condition)
            chart_file = self.chart_dir(model) / f"{artifact_stem}.png"
            chart_file.parent.mkdir(parents=True, exist_ok=True)
            chart_file.write_bytes(image_bytes)
            record["chart_path"] = build_static_path(chart_file)
        except Exception as exc:
            record["latency_plot_s"] = round(time.perf_counter() - plot_started, 4)
            record["error"] = str(exc)
            return record

        interp_started = time.perf_counter()
        try:
            response = self.context.client.post(
                f"{self.context.backend_url}/api/chat",
                data={
                    "message": case["question"],
                    "model": model,
                    "kg_disabled": str(kg_disabled).lower(),
                },
                files=[("images", ("chart.png", image_bytes, "image/png"))],
                headers=headers,
            )
            record["latency_interp_s"] = round(time.perf_counter() - interp_started, 4)
            record["latency_total_s"] = round(
                float(record["latency_plot_s"] or 0.0) + float(record["latency_interp_s"] or 0.0),
                4,
            )
            if response.status_code != 200:
                record["error"] = response.text
                return record
            payload = response.json()
            model_response = str(payload.get("content") or "")
            artifact_stem = self._artifact_stem(index, case, condition)
            response_file = self.response_dir(model) / f"{artifact_stem}.txt"
            response_file.parent.mkdir(parents=True, exist_ok=True)
            response_file.write_text(model_response, encoding="utf-8")
            record["model_response"] = model_response
            record["response_path"] = build_static_path(response_file)
        except Exception as exc:
            record["latency_interp_s"] = round(time.perf_counter() - interp_started, 4)
            record["latency_total_s"] = round(
                float(record["latency_plot_s"] or 0.0) + float(record["latency_interp_s"] or 0.0),
                4,
            )
            record["error"] = str(exc)
        return record

    def run_model(self, model: str) -> List[Dict[str, Any]]:
        return [
            self.evaluate_case(case, model=model, index=index, kg_disabled=False)
            for index, case in enumerate(
                tqdm(CHART_INTERP_CASES, desc=f"{self.task_name}:{model}", unit="case"),
                start=1,
            )
        ]


class KGAblationExperiment(BaseExperiment):
    task_name = "kg_ablation"

    def __init__(self, context: ExperimentContext, oracle: DataOracle):
        super().__init__(context, oracle)
        self.chart_experiment = ChartInterpretationExperiment(context, oracle)

    def run_model(self, model: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        conditions = [("kg_on", False), ("kg_off", True)]
        for condition, kg_disabled in conditions:
            for index, case in enumerate(
                tqdm(
                    KG_ABLATION_CHART_CASES,
                    desc=f"{self.task_name}:{model}:{condition}:chart",
                    unit="case",
                ),
                start=1,
            ):
                expected = self.oracle.safe_limit_expected(case)
                chart_record = self.chart_experiment.evaluate_case(
                    case,
                    model=model,
                    index=index,
                    kg_disabled=kg_disabled,
                    condition=condition,
                    task_name=self.task_name,
                )
                chart_record["subtask"] = "chart_interp"
                chart_record.update(expected)
                records.append(chart_record)
        return records


def build_experiment(task: str, context: ExperimentContext, oracle: DataOracle) -> BaseExperiment:
    mapping = {
        "nl_sql": NLToSQLExperiment,
        "display_cards": DisplayCardsExperiment,
        "plot_gen": PlotGenerationExperiment,
        "chart_interp": ChartInterpretationExperiment,
        "kg_ablation": KGAblationExperiment,
    }
    return mapping[task](context, oracle)


@app.command()
def main(
    task: str = typer.Option(..., help="One of: nl_sql, display_cards, plot_gen, chart_interp, kg_ablation."),
    models: Optional[str] = typer.Option(
        None,
        help="Comma-separated model names or menu numbers. If omitted in a terminal, prompts for model selection.",
    ),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Directory where JSONL results are written."),
    backend_url: str = typer.Option("http://localhost:8000", help="Base URL for the already-running backend."),
) -> None:
    normalized_task = task.strip().lower()
    if normalized_task not in TASK_NAMES:
        raise typer.BadParameter(f"Unsupported task '{task}'. Expected one of: {', '.join(sorted(TASK_NAMES))}")

    resolved_backend_url = backend_url.rstrip("/")
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=180.0) as client:
        selected_models = parse_models(models, client, resolved_backend_url)
        context = ExperimentContext(
            backend_url=resolved_backend_url,
            output_dir=resolved_output_dir,
            client=client,
            dataframe=load_merged_dataframe(),
            run_timestamp=utc_now_slug(),
        )
        oracle = DataOracle(context.dataframe)
        experiment = build_experiment(normalized_task, context, oracle)

        all_records: List[Dict[str, Any]] = []
        for model in selected_models:
            model_records = experiment.run_model(model)
            write_jsonl(experiment.result_path(model), model_records)
            all_records.extend(model_records)

    print_summary_table(all_records)


if __name__ == "__main__":
    app()
