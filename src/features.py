# src/features.py
import pandas as pd
import numpy as np

class CustomerFeatureBuilder:

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔹 [Features] Starting feature engineering...")

        # 1️⃣ Group by User_Id and aggregate features
        print("🔸 [Features] Aggregating customer-level features...")
        features = df.groupby("User_Id").agg(
            recency_days=("Customer_Age", "min"),
            transaction_count=("Trx_Vlu", "count"),
            total_transaction_value=("Trx_Vlu", "sum"),
            average_transaction_value=("Trx_Vlu", "mean"),
            total_points_used=("Points", "sum"),
            unique_merchants=("Mer_Id", "nunique"),
            unique_categories=("Category In English", "nunique")
        ).reset_index()
        print("✅ [Features] Aggregation completed.")
        print(f"📌 [Features] Number of customers: {features.shape[0]}")

        # 2️⃣ Log transform skewed columns
        print("🔸 [Features] Applying log1p transform on skewed columns...")
        for col in [
            "transaction_count",
            "total_transaction_value",
            "total_points_used"
        ]:
            features[f"log_{col}"] = np.log1p(features[col])
        print("✅ [Features] Log transformation completed.")

        print("🔹 [Features] Feature engineering finished successfully.\n")
        return features
