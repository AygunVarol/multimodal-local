
import os, glob, re
import duckdb
import pandas as pd
from typing import Tuple, List

DATA_DIR = os.getenv("DATA_DIR", "../data")
CSV_PATH = os.getenv("CSV_PATH")  # optional single-file override

def _read_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass
    if "location" not in df.columns:
        fname = os.path.basename(path).lower()
        m = re.match(r"(\w+)_measurements_iaq_1h_avg\.csv", fname)
        loc = m.group(1) if m else "unknown"
        df["location"] = loc
    df["location"] = df["location"].astype(str).str.lower()
    return df

def load_dataframe() -> pd.DataFrame:
    if CSV_PATH:
        return _read_one(CSV_PATH)
    pattern = os.path.join(DATA_DIR, "*_measurements_IAQ_1h_avg.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        off = os.path.join(DATA_DIR, "office_measurements_IAQ_1h_avg.csv")
        if os.path.exists(off):
            files = [off]
    if not files:
        raise FileNotFoundError(f"No IAQ CSV files found under {DATA_DIR}")
    frames = [ _read_one(p) for p in files ]
    all_cols: List[str] = sorted(set().union(*[ set(f.columns) for f in frames ]))
    frames = [ f.reindex(columns=all_cols) for f in frames ]
    df = pd.concat(frames, ignore_index=True)
    return df

def table_schema(df: pd.DataFrame) -> str:
    return "\n".join([f"{c} ({str(t)})" for c,t in zip(df.columns, df.dtypes)])

class DuckDBRunner:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.con = duckdb.connect()
        self.con.register("iaq", self.df)

    def safe_execute(self, sql: str) -> Tuple[list[str], list[list]]:
        sql_strip = sql.strip().lower()
        if not (sql_strip.startswith("select") or sql_strip.startswith("with")):
            raise ValueError("Only SELECT statements are allowed.")
        for k in ["insert","update","delete","drop","alter","create","attach","copy","pragma"]:
            if k in sql_strip:
                raise ValueError("Unsafe SQL detected.")
        df = self.con.execute(sql).df()
        return list(df.columns), df.values.tolist()
