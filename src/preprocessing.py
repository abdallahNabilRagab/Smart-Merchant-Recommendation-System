# src/preprocessing.py 
import pandas as pd
from pathlib import Path
from src.utils import assert_columns_exist


class TransactionLoader:

    REQUIRED_COLUMNS = [
        "User_Id",
        "Mer_Id",
        "Trx_Vlu",
        "Points",
        "Customer_Age",
        "Trx_Age",
        "Category In English"
    ]

    def __init__(self, file_path: Path):
        self.file_path = file_path
        print(f"🔹 [Preprocessing] TransactionLoader initialized with file: {self.file_path}")

    def load(self) -> pd.DataFrame:
        print("🔸 [Preprocessing] Loading data...")

        df = pd.read_parquet(self.file_path)

        print("✅ [Preprocessing] Data loaded successfully.")
        print(f"📌 [Preprocessing] Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        print("🔸 [Preprocessing] Validating required columns...")
        assert_columns_exist(df, self.REQUIRED_COLUMNS)
        print("✅ [Preprocessing] Required columns are present.")

        print("🔸 [Preprocessing] Converting 'Category In English' to categorical dtype...")
        df["Category In English"] = df["Category In English"].astype("category")
        print("✅ [Preprocessing] Conversion completed.")

        print("🔹 [Preprocessing] Data preprocessing finished.\n")
        return df
