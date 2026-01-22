# src/split.py
import pandas as pd
from src.config import TRANSACTIONS_PARQUET_PATH, DATA_RAW


def per_user_time_split(test_size: float = 0.3):
    """
    Split transactions per user using Trx_Age.

    - Train = older transactions
    - Test = newer transactions
    """

    print("🔹 [Split] Loading transactions for splitting...")
    df = pd.read_parquet(TRANSACTIONS_PARQUET_PATH)
    print(f"✅ [Split] Loaded transactions: {df.shape[0]} rows")

    # Sort by user + Trx_Age (oldest first)
    print("🔸 [Split] Sorting by User_Id and Trx_Age (oldest first)...")
    df = df.sort_values(["User_Id", "Trx_Age"], ascending=[True, False])

    train_list = []
    test_list = []

    print("🔸 [Split] Splitting per user...")
    for user_id, user_df in df.groupby("User_Id"):
        split_index = int(len(user_df) * (1 - test_size))

        train_list.append(user_df.iloc[:split_index])
        test_list.append(user_df.iloc[split_index:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Save paths
    train_path = DATA_RAW / "transactions_train.parquet"
    test_path = DATA_RAW / "transactions_test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("✅ [Split] Split completed successfully.")
    print(f"📌 [Split] Train rows: {train_df.shape[0]}")
    print(f"📌 [Split] Test rows: {test_df.shape[0]}")
    print(f"📌 [Split] Saved train: {train_path}")
    print(f"📌 [Split] Saved test: {test_path}\n")

    return train_path, test_path
