
from fastapi import APIRouter, Header, HTTPException
from ..services.kg_store import KGStore
from ..services.duckdb_engine import load_dataframe
import pandas as pd
from typing import Optional

router = APIRouter()
kg = KGStore()
_df = load_dataframe()
_df["Date"] = pd.to_datetime(_df["Date"])

def _header_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def status_for(metric: str, value: float, kg_disabled: bool = False):
    if kg_disabled:
        return "unknown"
    label = {"temperature_c":"temperature", "humidity_rh":"humidity"}.get(metric)
    if not label: return "info"
    rr = kg.recommended_range(label)
    if not rr or "min" not in rr or "max" not in rr: return "info"
    if value < rr["min"]: return "warn-low"
    if value > rr["max"]: return "warn-high"
    return "ok"

@router.get("/display/cards")
def display_cards(
    date: str,
    location: str = "office",
    kg_disabled: bool = False,
    x_kg_disabled: Optional[str] = Header(None, alias="X-KG-Disabled"),
):
    try:
        d0 = pd.to_datetime(date)
    except Exception:
        raise HTTPException(400, "Use date like 2025-08-19")
    kg_disabled = kg_disabled or _header_truthy(x_kg_disabled)
    d1 = d0 + pd.Timedelta(days=1)
    df = _df[(_df["Date"] >= d0) & (_df["Date"] < d1)]
    df = df[df["location"].str.lower().str.contains(location.lower())]
    if df.empty:
        raise HTTPException(404, "No data for that date/location.")
    cards = []
    for metric in ["temperature_c", "humidity_rh"]:
        mean_val = float(df[metric].mean())
        st = status_for(metric, mean_val, kg_disabled=kg_disabled)
        cards.append({"title": metric, "value": round(mean_val, 2), "status": st})
    ticker = " • ".join([f"{c['title']}: {c['value']}" for c in cards])
    return {"date": date, "location": location, "cards": cards, "ticker": ticker}
