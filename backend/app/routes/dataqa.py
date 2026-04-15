
from fastapi import APIRouter, Header, HTTPException
from ..models.schemas import DataAskRequest, DataAskResponse
from ..services.duckdb_engine import load_dataframe, table_schema, DuckDBRunner
from ..services.ollama_client import generate
from ..services.kg_store import KGStore
from io import BytesIO
import base64, re
from statistics import mean
from typing import Any, Dict, List, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

router = APIRouter()
_df = load_dataframe()
_runner = DuckDBRunner(_df)
kg_store = KGStore()

KNOWN_LOCATIONS_HINT = (
    "Locations are in column 'location' (values include kitchen, hallway, office). "
    "Only add a location WHERE clause when the question asks for a specific location; "
    "for kitchen, use WHERE lower(location) LIKE '%kitchen%'. \n"
)
SYSTEM_SQL = (
    "You are a data assistant that writes a single valid DuckDB SQL query to answer a question. "
    "Only output the SQL. Do not include explanations or code fences."
)
COMFORT_CONTEXT = (
    "Comfort context: recommended temperature range is 20-26 C and recommended humidity "
    "range is 30-60% when those metrics are relevant."
)

METRIC_LABEL_LOOKUP = {
    "temperature_c": "temperature",
    "humidity_rh": "humidity",
    "pressure_hpa": "pressure",
    "IAQ_proxy": "iaq score",
    "gas_resistance_ohms": "gas resistance"
}

def _extract_sql(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()
        else:
            cleaned = cleaned.replace("```", "").strip()
    lines = cleaned.splitlines()
    if lines and re.fullmatch(r"\s*sql\s*", lines[0], flags=re.IGNORECASE):
        cleaned = "\n".join(lines[1:]).strip()
    cleaned = re.sub(r"^\s*(sql|query)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    starts_ok = re.match(r"^\s*(select|with)\b", cleaned, flags=re.IGNORECASE)
    if not starts_ok:
        match = re.search(r"\bselect\b", cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[match.start():]
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0]
    return cleaned.strip()

def _header_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _kg_prompt_context(question: str, kg_disabled: bool) -> str:
    if kg_disabled:
        return ""
    snippets = kg_store.short_context_for(question)
    parts = [COMFORT_CONTEXT]
    if snippets:
        parts.append(f"Relevant KG facts:\n{snippets}")
    return "\n".join(parts)

def _pervasive_status_for_results(columns: List[str], rows: List[List[Any]], parsed: Optional[Dict[str, Any]]):
    if not columns or not rows:
        return None
    statuses = []
    loc_raw = (parsed or {}).get("location")
    loc_title = loc_raw.title() if isinstance(loc_raw, str) else None
    for idx, col in enumerate(columns):
        metric_label = METRIC_LABEL_LOOKUP.get(col)
        if not metric_label:
            continue
        numeric_values: List[float] = []
        for row in rows:
            if idx >= len(row):
                continue
            val = row[idx]
            if val is None:
                continue
            try:
                numeric_values.append(float(val))
            except (TypeError, ValueError):
                continue
        if not numeric_values:
            continue
        avg_val = mean(numeric_values)
        status = "info"
        rec = kg_store.recommended_range(metric_label)
        rec_min = rec.get("min") if rec and "min" in rec else None
        rec_max = rec.get("max") if rec and "max" in rec else None
        if rec_min is not None and avg_val < rec_min:
            status = "warn-low"
        elif rec_max is not None and avg_val > rec_max:
            status = "warn-high"
        elif rec is not None:
            status = "ok"
        statuses.append({
            "metric": col,
            "status": status,
            "mean_value": round(avg_val, 3),
            "recommended_min": rec_min,
            "recommended_max": rec_max,
            "location": loc_title
        })
    return statuses or None

@router.post("/data/ask", response_model=DataAskResponse)
def ask_data(req: DataAskRequest, x_kg_disabled: Optional[str] = Header(None, alias="X-KG-Disabled")):
    schema = table_schema(_df)
    sample = _df.head(5).to_dict(orient="records")
    kg_disabled = req.kg_disabled or _header_truthy(x_kg_disabled)
    kg_context = _kg_prompt_context(req.question, kg_disabled)
    prompt_parts = [
        f"{SYSTEM_SQL}\n",
        f"{KNOWN_LOCATIONS_HINT}",
        "Table name: iaq\n",
        f"Schema (name (dtype)):\n{schema}\n\n",
        f"Sample rows (JSON):\n{sample}\n\n",
    ]
    if kg_context:
        prompt_parts.append(f"{kg_context}\n\n")
    prompt_parts.append(
        f"Question: {req.question}\n\n"
        f"Return ONLY one DuckDB SQL statement that starts with SELECT and uses table iaq."
    )
    prompt = "".join(prompt_parts)
    gen = generate(prompt, model=req.model)
    sql = _extract_sql((gen.get("content") or "").strip().strip("`"))
    try:
        cols, rows = _runner.safe_execute(sql)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    parsed = _parse_question(req.question)
    statuses = _pervasive_status_for_results(cols, rows, parsed)
    return DataAskResponse(
        sql=sql,
        columns=cols,
        rows=rows,
        note="Executed with DuckDB over in-memory table 'iaq'.",
        pervasive_status=statuses
    )

# --- Plotting helpers ---
def _parse_question(q: str):
    ql = q.lower()
    metric_map = {
        "temperature": "temperature_c",
        "temp": "temperature_c",
        "humidity": "humidity_rh",
        "pressure": "pressure_hpa",
        "iaq": "IAQ_proxy",
        "iaq score": "IAQ_proxy",
        "gas resistance": "gas_resistance_ohms"
    }
    metric = None
    for k, v in metric_map.items():
        if re.search(rf"\b{k}\b", ql):
            metric = v; break
    dates = re.findall(r"(20\d{2}-\d{2}-\d{2})", q)
    date_from = dates[0] if dates else None
    date_to = dates[1] if len(dates) > 1 else None
    loc = None
    for cand in ["kitchen","office","hallway"]:
        if cand in ql:
            loc = cand
            break
    return {"metric": metric, "date_from": date_from, "date_to": date_to, "location": loc}

@router.post("/data/plot")
def plot_question(req: DataAskRequest, x_kg_disabled: Optional[str] = Header(None, alias="X-KG-Disabled")):
    parsed = _parse_question(req.question)
    kg_disabled = req.kg_disabled or _header_truthy(x_kg_disabled)
    metric = parsed["metric"]
    if not metric:
        raise HTTPException(400, "Could not detect metric (try: temperature, humidity, pressure, IAQ).")
    df = _df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if parsed["date_from"]:
        d0 = pd.to_datetime(parsed["date_from"])
        d1 = pd.to_datetime(parsed["date_to"]) if parsed["date_to"] else d0 + pd.Timedelta(days=1)
        df = df[(df["Date"] >= d0) & (df["Date"] < d1)]
    if parsed["location"]:
        df = df[df["location"].str.lower().str.contains(parsed["location"])]
    if df.empty:
        raise HTTPException(404, "No data for your filters.")
    df = df.sort_values("Date")
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["Date"], df[metric], marker="o", linewidth=1.2)
    ax.set_ylabel(metric)
    locator = mdates.HourLocator(interval=1)
    ax.xaxis.set_major_locator(locator)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)
    loc_label = parsed["location"].title() if parsed["location"] else None
    if parsed["date_from"] and not parsed["date_to"]:
        ax.set_xlabel(f"time on {parsed['date_from']}")
    else:
        ax.set_xlabel("time")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    metric_label_map = {
        "temperature_c": "Temperature",
        "humidity_rh": "Humidity",
        "pressure_hpa": "Pressure",
        "IAQ_proxy": "IAQ score"
    }
    metric_title = metric_label_map.get(metric, metric)
    if parsed["date_from"] and not parsed["date_to"]:
        base_title = f"{metric_title} on {parsed['date_from']}"
    elif parsed["date_from"] and parsed["date_to"]:
        base_title = f"{metric_title} from {parsed['date_from']} to {parsed['date_to']}"
    else:
        base_title = metric_title
    if loc_label:
        base_title += f" at {loc_label}"
    ax.set_title(base_title)
    lookup = {"temperature_c":"temperature","humidity_rh":"humidity"}
    label = lookup.get(metric)
    if label and not kg_disabled:
        r = kg_store.recommended_range(label)
        if r and "min" in r and "max" in r:
            ax.axhspan(r["min"], r["max"], alpha=0.1)
    buf = BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=160); plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    sql = f"SELECT Date, {metric} FROM iaq"
    return {"image_b64": b64, "metric": metric, "sql": sql, "parsed": parsed}

@router.post("/data/value")
def value_question(req: DataAskRequest):
    parsed = _parse_question(req.question)
    metric = parsed["metric"]
    if not metric:
        raise HTTPException(400, "Could not detect metric (try: temperature, humidity, pressure, IAQ).")
    if not parsed["date_from"]:
        raise HTTPException(400, "Please provide a date like 2025-08-19.")
    df = _df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d0 = pd.to_datetime(parsed["date_from"]); d1 = pd.to_datetime(parsed["date_to"]) if parsed["date_to"] else d0 + pd.Timedelta(days=1)
    dsel = df[(df["Date"] >= d0) & (df["Date"] < d1)]
    if parsed["location"]:
        dsel = dsel[dsel["location"].str.lower().str.contains(parsed["location"])]
    if dsel.empty:
        raise HTTPException(404, "No data for your filters.")
    val = float(dsel[metric].mean())
    return {"metric": metric, "date": parsed["date_from"], "value_mean": val}
