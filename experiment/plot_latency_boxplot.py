from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# python experiment/plot_latency_boxplot.py --rename 'gemma3_4b-it-qat=Gemma 3 4B,ministral-3_3b=Ministral 3 3B,granite3.2-vision_latest=Granite 3 2B,qwen3-vl_4b=Qwen 3 4B'

SAFETY_RESULT_FILENAMES = ("raw_safety_experiment.csv", "raw_safety_eval.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side latency boxplot for all models in a results run."
    )
    parser.add_argument(
        "--run_dir",
        default="results/20260203_171132_granite",
        help="Path to a results run directory (contains model subfolders).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file path (default: <run_dir>/latency_boxplot.png).",
    )
    parser.add_argument(
        "--rename",
        default=None,
        help=(
            "Optional comma-separated renames in the form "
            "'folder=Label,other_folder=Other Label'."
        ),
    )
    return parser.parse_args()


def discover_model_runs(run_dir: Path) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        for filename in SAFETY_RESULT_FILENAMES:
            csv_path = child / filename
            if csv_path.exists():
                entries.append((child.name, csv_path))
                break
    return entries


def load_latencies(csv_path: Path) -> List[float]:
    latencies: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "latency_s" not in (reader.fieldnames or []):
            raise ValueError(f"Missing latency_s column in {csv_path}")
        for row in reader:
            raw = (row.get("latency_s") or "").strip()
            if not raw:
                continue
            try:
                latencies.append(float(raw))
            except ValueError:
                continue
    return latencies


def prettify_model_name(folder_name: str) -> str:
    return folder_name.replace("_", ":")


def parse_renames(raw: str | None) -> dict:
    if not raw:
        return {}
    mapping: dict = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            mapping[key] = value
    return mapping


def resolve_fontsize(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return font_manager.FontProperties(size=value).get_size_in_points()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_runs = discover_model_runs(run_dir)
    if not model_runs:
        raise RuntimeError(f"No model runs found under {run_dir}")

    renames = parse_renames(args.rename)

    labels: List[str] = []
    data: List[List[float]] = []
    for folder_name, csv_path in model_runs:
        latencies = load_latencies(csv_path)
        if not latencies:
            continue
        label = renames.get(folder_name, prettify_model_name(folder_name))
        labels.append(label)
        data.append(latencies)

    if not data:
        raise RuntimeError(f"No latency data found under {run_dir}")

    out_path = Path(args.out) if args.out else run_dir / "latency_boxplot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = max(6.0, 1.6 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 4))
    boxplot_kwargs = {
        "showfliers": False,
        "patch_artist": False,
        "boxprops": {"linewidth": 1.5},
        "whiskerprops": {"linewidth": 1.5},
        "capprops": {"linewidth": 1.5},
        "medianprops": {"color": "red", "linewidth": 1.5},
    }
    try:
        ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(data, labels=labels, **boxplot_kwargs)
    scale = 1.25
    label_size = resolve_fontsize(plt.rcParams["axes.labelsize"]) * scale
    ax.set_ylabel("Latency (s)", fontsize=label_size)
    #ax.set_title("Latency by model")
    x_tick_size = resolve_fontsize(plt.rcParams["xtick.labelsize"]) * scale
    y_tick_size = resolve_fontsize(plt.rcParams["ytick.labelsize"]) * scale
    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=x_tick_size,
    )
    ax.tick_params(axis="y", labelsize=y_tick_size)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved latency boxplot to: {out_path}")


if __name__ == "__main__":
    main()
