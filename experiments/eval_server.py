"""Standalone evaluation server for experiment result review."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
EVAL_UI_DIR = EXPERIMENTS_DIR / "eval_ui"
KNOWN_TASK_PREFIXES = ["nl_sql", "display_cards", "plot_gen", "chart_interp", "kg_ablation"]

app = FastAPI(title="Experiment Evaluation Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(EXPERIMENTS_DIR)), name="static")


class GradeRequest(BaseModel):
    file: str
    index: int
    correct: bool
    note: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def task_type_for_file(path: Path) -> str:
    name = path.name
    for prefix in KNOWN_TASK_PREFIXES:
        if name.startswith(f"{prefix}_"):
            return prefix
    return "other"


def safe_result_path(file_name: str) -> Path:
    path = RESULTS_DIR / Path(file_name).name
    if not path.exists() or path.suffix != ".jsonl":
        raise HTTPException(status_code=404, detail=f"Result file '{file_name}' was not found.")
    return path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def latency_for_record(record: Dict[str, Any]) -> Optional[float]:
    for key in ("latency_total_s", "latency_s", "latency_interp_s", "latency_plot_s"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def task_label(record: Dict[str, Any]) -> str:
    task = str(record.get("task") or "unknown")
    condition = record.get("condition")
    if condition:
        return f"{task} [{condition}]"
    return task


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        model = str(record.get("model") or "default")
        task = task_label(record)
        bucket = grouped.setdefault(
            (model, task),
            {
                "model": model,
                "task": task,
                "correct": 0,
                "graded": 0,
                "pending": 0,
                "total": 0,
                "latencies": [],
            },
        )
        bucket["total"] += 1
        if record.get("correct") is True:
            bucket["correct"] += 1
            bucket["graded"] += 1
        elif record.get("correct") is False:
            bucket["graded"] += 1
        else:
            bucket["pending"] += 1
        latency = latency_for_record(record)
        if latency is not None:
            bucket["latencies"].append(latency)

    rows: List[Dict[str, Any]] = []
    for _, bucket in sorted(grouped.items()):
        accuracy = None
        if bucket["graded"] > 0:
            accuracy = bucket["correct"] / bucket["graded"]
        mean_latency = None
        if bucket["latencies"]:
            mean_latency = sum(bucket["latencies"]) / len(bucket["latencies"])
        rows.append(
            {
                "model": bucket["model"],
                "task": bucket["task"],
                "correct": bucket["correct"],
                "graded": bucket["graded"],
                "pending": bucket["pending"],
                "total": bucket["total"],
                "accuracy": accuracy,
                "mean_latency_s": mean_latency,
            }
        )
    return rows


@app.get("/")
def index() -> FileResponse:
    return FileResponse(EVAL_UI_DIR / "index.html")


@app.get("/api/results")
def list_results() -> List[Dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        records = load_jsonl(path)
        graded = sum(1 for record in records if record.get("correct") is not None)
        correct = sum(1 for record in records if record.get("correct") is True)
        items.append(
            {
                "file": path.name,
                "task_type": task_type_for_file(path),
                "total": len(records),
                "graded": graded,
                "pending": len(records) - graded,
                "correct": correct,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return items


@app.get("/api/results/{file_name}")
def get_results(file_name: str) -> List[Dict[str, Any]]:
    path = safe_result_path(file_name)
    return load_jsonl(path)


@app.post("/api/grade")
def grade_record(request: GradeRequest) -> Dict[str, Any]:
    path = safe_result_path(request.file)
    records = load_jsonl(path)
    if request.index < 0 or request.index >= len(records):
        raise HTTPException(status_code=400, detail="Record index is out of range.")
    record = records[request.index]
    record["correct"] = request.correct
    if request.note and request.note.strip():
        record["note"] = request.note.strip()
    else:
        record.pop("note", None)
    record["graded_at"] = utc_now_iso()
    records[request.index] = record
    write_jsonl(path, records)
    return {"status": "ok", "record": record}


@app.get("/api/summary")
def summary() -> Dict[str, Any]:
    all_records: List[Dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        all_records.extend(load_jsonl(path))
    return {"generated_at": utc_now_iso(), "rows": summarize_records(all_records)}
