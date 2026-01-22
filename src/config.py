
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "Data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "Data" / "clean"
MODELS = PROJECT_ROOT / "models"

RANDOM_STATE = 42


# -------------------------------
# Convert CSV to parquet (once)
# -------------------------------
def ensure_transactions_parquet():
    csv_file = DATA_RAW / "Cleaned_Data_Merchant_Level_2.csv"
    parquet_file = DATA_RAW / "transactions.parquet"

    if not parquet_file.exists():
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)
            print("✅ Converted CSV to transactions.parquet successfully")
        else:
            raise FileNotFoundError(
                f"CSV file not found: {csv_file}"
            )
    else:
        print("ℹ️ transactions.parquet already exists. Skipping conversion.")

    return parquet_file


# Run conversion when config is imported
TRANSACTIONS_PARQUET_PATH = ensure_transactions_parquet()


# =========================
# New: train/test paths
# =========================
TRANSACTIONS_TRAIN_PATH = DATA_RAW / "transactions_train.parquet"
TRANSACTIONS_TEST_PATH = DATA_RAW / "transactions_test.parquet"
