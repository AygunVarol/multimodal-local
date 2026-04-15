from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXPERIMENT_NAME = "safety-experiment"
MODEL_MENU = [
    ("Ministral 3 3B", "ministral-3:3b"),
    ("Qwen 3 VL 4B", "qwen3-vl:4b"),
    ("Gemma 3 4B IT QAT", "gemma3:4b-it-qat"),
    ("Granite 3.2 Vision", "granite3.2-vision:latest"),
]

SYSTEM_PROMPT = "Check for fire, smoke indoor environment to evaluate safety."
USER_PROMPT = "Is this place safe? Answer with EXACTLY ONE WORD: SAFE or UNSAFE."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the safety-experiment image classification evaluation.")
    parser.add_argument("--safe_dir", default="experiment/safe", help="Directory with safe images.")
    parser.add_argument("--unsafe_dir", default="experiment/unsafe", help="Directory with unsafe images.")
    parser.add_argument("--n_per_class", type=int, default=60, help="Number of images per class.")
    parser.add_argument("--seed", type=int, default=31, help="Random seed for sampling.") #42
    parser.add_argument("--out", default="results", help="Base output directory.")
    parser.add_argument(
        "--base_url",
        default=None,
        help="Optional override for OLLAMA_BASE_URL (default env or http://localhost:11434).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names or menu numbers to evaluate. "
            "If omitted in a terminal, prompts for model selection."
        ),
    )
    return parser.parse_args()


def menu_model_names() -> List[str]:
    return [model for _, model in MODEL_MENU]


def parse_models(raw: str | None, menu: List[str] | None = None) -> List[str]:
    if not raw:
        return []
    menu = menu or menu_model_names()
    value = raw.strip()
    if value.lower() in {"all", "*"}:
        return list(menu)
    parts = re.split(r"[,\s]+", value)
    models: List[str] = []
    seen = set()
    for part in parts:
        name = part.strip()
        if not name:
            continue
        if name.isdigit():
            index = int(name)
            if index < 1 or index > len(menu):
                raise ValueError(f"Model number {index} is outside 1-{len(menu)}.")
            model = menu[index - 1]
        else:
            model = name
        if model not in seen:
            models.append(model)
            seen.add(model)
    return models


def prompt_model_selection() -> List[str]:
    menu = menu_model_names()
    print("Select model(s) to evaluate:")
    for index, (label, model) in enumerate(MODEL_MENU, start=1):
        print(f"  {index}. {label} ({model})")
    print("Enter one number, comma-separated numbers, or 'all'.")
    while True:
        selection = input("Model selection [1]: ").strip() or "1"
        try:
            models = parse_models(selection, menu)
        except ValueError as exc:
            print(f"Invalid model selection: {exc}", file=sys.stderr)
            continue
        if models:
            return models
        print("Select at least one model.", file=sys.stderr)


def model_dir_name(model: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model.strip())
    return safe or "model"


def load_ollama_client(repo_root: Path):
    client_path = repo_root / "backend" / "app" / "services" / "ollama_client.py"
    if not client_path.exists():
        raise FileNotFoundError(f"ollama_client.py not found at {client_path}")
    spec = importlib.util.spec_from_file_location("ollama_client", client_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load ollama_client from {client_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_ollama(base_url: str, model: str) -> None:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Ollama not reachable at {url}: {exc}") from exc
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON from {url}") from exc

    models: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        if isinstance(data.get("models"), list):
            models = data.get("models", [])
    elif isinstance(data, list):
        models = data

    names = set()
    for item in models:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model")
            if name:
                names.add(name)

    if model not in names:
        available = ", ".join(sorted(names)) if names else "<none>"
        raise RuntimeError(f"Model '{model}' not found in Ollama tags. Available: {available}")


def collect_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return sorted(paths)


def sample_images(paths: List[Path], n: int, rng: random.Random, label: str) -> List[Path]:
    if n <= 0:
        raise ValueError("--n_per_class must be > 0")
    if len(paths) < n:
        print(
            f"Warning: {label} has only {len(paths)} images, sampling with replacement to reach {n}.",
            file=sys.stderr,
        )
        return rng.choices(paths, k=n)
    return rng.sample(paths, n)


def encode_image(path: Path) -> str:
    content = path.read_bytes()
    return base64.b64encode(content).decode("utf-8")


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_prediction(text: str) -> str:
    normalized = normalize_text(text or "")
    if not normalized:
        return "UNKNOWN"
    unsafe_markers = ["unsafe", "not safe", "danger", "fire", "smoke"]
    if any(marker in normalized for marker in unsafe_markers):
        return "UNSAFE"
    if "safe" in normalized:
        return "SAFE"
    return "UNKNOWN"


def percentile(sorted_vals: List[float], percent: float) -> float:
    if not sorted_vals:
        return float("nan")
    index = max(0, math.ceil(percent * len(sorted_vals)) - 1)
    return sorted_vals[index]


def save_confusion_matrix(path: Path, tp: int, fn: int, fp: int, tn: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[tp, fn], [fp, tn]]
    fig, ax = plt.subplots(figsize=(4, 3))
    vmax = max(max(row) for row in matrix) if matrix else 1
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(vmax, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SAFE", "UNSAFE"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SAFE", "UNSAFE"])
    threshold = vmax / 2.0 if vmax else 0.5
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_latency_boxplot(path: Path, latencies: List[float], label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.boxplot(latencies, vert=True)
    ax.set_ylabel("Latency (s)")
    ax.set_xticks([1])
    ax.set_xticklabels([label])
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_latex_table(path: Path, metrics: Dict[str, str]) -> None:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Accuracy & {metrics['accuracy']} \\\\",
        f"Safe Accuracy & {metrics['safe_accuracy']} \\\\",
        f"Unsafe Accuracy & {metrics['unsafe_accuracy']} \\\\",
        f"TP & {metrics['tp']} \\\\",
        f"TN & {metrics['tn']} \\\\",
        f"FP & {metrics['fp']} \\\\",
        f"FN & {metrics['fn']} \\\\",
        f"Latency Mean (s) & {metrics['latency_mean_s']} \\\\",
        f"Latency Median (s) & {metrics['latency_median_s']} \\\\",
        f"Latency P95 (s) & {metrics['latency_p95_s']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_model(
    model: str,
    samples: List[Tuple[Path, str]],
    out_root: Path,
    ollama_client: Any,
) -> None:
    warmup_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello how are you?"},
    ]
    try:
        ollama_client.chat(warmup_messages, model=model)
        print(f"[{model}] Warm-up completed.")
    except Exception as exc:
        print(f"[{model}] Warm-up failed: {exc}")

    figs_dir = out_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for idx, (path, true_label) in enumerate(samples, start=1):
        image_b64 = encode_image(path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
        start = time.perf_counter()
        model_text = ""
        try:
            result = ollama_client.chat(messages, images=[image_b64], model=model)
            model_text = result.get("content") or ""
        except Exception as exc:
            model_text = f"ERROR: {exc}"
        latency = time.perf_counter() - start
        pred_label = parse_prediction(model_text)
        is_correct = pred_label == true_label
        latencies.append(latency)
        rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": is_correct,
                "latency_s": latency,
                "model_text": model_text,
                "image_path": str(path),
            }
        )
        print(f"[{model}] [{idx}/{len(samples)}] {path.name} -> {pred_label} ({latency:.3f}s)")

    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    accuracy = correct / total if total else 0.0

    safe_total = sum(1 for row in rows if row["true_label"] == "SAFE")
    unsafe_total = sum(1 for row in rows if row["true_label"] == "UNSAFE")
    safe_correct = sum(1 for row in rows if row["true_label"] == "SAFE" and row["pred_label"] == "SAFE")
    unsafe_correct = sum(1 for row in rows if row["true_label"] == "UNSAFE" and row["pred_label"] == "UNSAFE")

    safe_accuracy = safe_correct / safe_total if safe_total else 0.0
    unsafe_accuracy = unsafe_correct / unsafe_total if unsafe_total else 0.0

    tp = sum(1 for row in rows if row["true_label"] == "SAFE" and row["pred_label"] == "SAFE")
    fn = sum(1 for row in rows if row["true_label"] == "SAFE" and row["pred_label"] != "SAFE")
    tn = sum(1 for row in rows if row["true_label"] == "UNSAFE" and row["pred_label"] == "UNSAFE")
    fp = sum(1 for row in rows if row["true_label"] == "UNSAFE" and row["pred_label"] != "UNSAFE")

    lat_sorted = sorted(latencies)
    latency_mean = statistics.mean(lat_sorted) if lat_sorted else 0.0
    latency_min = lat_sorted[0] if lat_sorted else 0.0
    latency_max = lat_sorted[-1] if lat_sorted else 0.0
    latency_median = statistics.median(lat_sorted) if lat_sorted else 0.0
    latency_p95 = percentile(lat_sorted, 0.95)

    raw_path = out_root / "raw_safety_experiment.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["true_label", "pred_label", "is_correct", "latency_s", "model_text", "image_path"],
        )
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["latency_s"] = f"{row['latency_s']:.6f}"
            writer.writerow(row_out)

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    summary_metrics = {
        "accuracy": fmt(accuracy),
        "safe_accuracy": fmt(safe_accuracy),
        "unsafe_accuracy": fmt(unsafe_accuracy),
        "tp": str(tp),
        "tn": str(tn),
        "fp": str(fp),
        "fn": str(fn),
        "latency_mean_s": fmt(latency_mean),
        "latency_min_s": fmt(latency_min),
        "latency_max_s": fmt(latency_max),
        "latency_median_s": fmt(latency_median),
        "latency_p95_s": fmt(latency_p95),
    }

    summary_path = out_root / "summary_safety_experiment.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_metrics.keys()))
        writer.writeheader()
        writer.writerow(summary_metrics)

    write_latex_table(out_root / "table_safety_experiment.tex", summary_metrics)
    save_confusion_matrix(figs_dir / "confusion_matrix.png", tp, fn, fp, tn)
    save_latency_boxplot(figs_dir / "latency_boxplot.png", latencies, model)

    print(f"Results written to: {out_root}")


def main() -> None:
    args = parse_args()
    if args.base_url:
        os.environ["OLLAMA_BASE_URL"] = args.base_url

    repo_root = Path(__file__).resolve().parents[1]
    ollama_client = load_ollama_client(repo_root)

    base_url = getattr(ollama_client, "OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    try:
        models = parse_models(args.models)
    except ValueError as exc:
        raise SystemExit(f"Invalid --models value: {exc}") from exc
    if not models and sys.stdin.isatty():
        models = prompt_model_selection()
    if not models:
        if hasattr(ollama_client, "available_models"):
            try:
                models = list(ollama_client.available_models())
            except Exception:
                models = []
    if not models:
        models = parse_models(os.getenv("OLLAMA_MODELS"))
    if not models:
        models = [getattr(ollama_client, "OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat"))]

    for model in models:
        check_ollama(base_url, model)

    rng = random.Random(args.seed)

    safe_dir = Path(args.safe_dir).expanduser()
    unsafe_dir = Path(args.unsafe_dir).expanduser()
    safe_paths = collect_images(safe_dir)
    unsafe_paths = collect_images(unsafe_dir)

    safe_samples = sample_images(safe_paths, args.n_per_class, rng, "safe")
    unsafe_samples = sample_images(unsafe_paths, args.n_per_class, rng, "unsafe")

    samples: List[Tuple[Path, str]] = [(path, "SAFE") for path in safe_samples] + [
        (path, "UNSAFE") for path in unsafe_samples
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out) / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    sampled = {
        "experiment": EXPERIMENT_NAME,
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "safe": [str(path) for path in safe_samples],
        "unsafe": [str(path) for path in unsafe_samples],
    }
    (run_root / "sampled_images.json").write_text(json.dumps(sampled, indent=2), encoding="utf-8")

    for model in models:
        out_root = run_root / model_dir_name(model)
        evaluate_model(model, samples, out_root, ollama_client)


if __name__ == "__main__":
    main()
