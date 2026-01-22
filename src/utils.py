# src/utils.py
from pathlib import Path
import pandas as pd
import joblib
import logging

def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Path):
    return joblib.load(path)

def assert_columns_exist(df: pd.DataFrame, cols: list):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
