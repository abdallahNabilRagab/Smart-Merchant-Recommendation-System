# src/recommendation.py

import pandas as pd
from pathlib import Path

class MerchantRecommender:
    """
    Hybrid Merchant Recommendation System:
    - User recent behavior
    - Cluster-level popularity

    Enrichment:
    - Explainable recommendations
    - Auto-save recommendations to Data/clean/recommendations.parquet
    """

    def __init__(self, min_recency_days: int = 7):
        self.min_recency_days = min_recency_days
        self.save_path = Path("Data/clean/recommendations.parquet")
        print(
            f"🔹 [Recommendation] MerchantRecommender initialized "
            f"with min_recency_days={self.min_recency_days}"
        )

    def recommend(
        self,
        customer_clusters: pd.DataFrame,
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate merchant recommendations with explainability and save automatically.

        Returns
        -------
        DataFrame with:
        - User_Id
        - cluster_id
        - recommendation_type
        - top_merchants (list of merchant IDs)
        - recommendations (list of dicts with merchant_id + reason)
        """

        print("🔸 [Recommendation] Starting recommendation generation...")

        # -------------------------------------------------
        # Step 1: Aggregate user-merchant interaction
        # -------------------------------------------------
        user_merchant = (
            transactions
            .groupby(["User_Id", "Mer_Id"], as_index=False)
            .agg(
                Points=("Points", "sum"),
                Trx_Age=("Trx_Age", "min")
            )
        )

        # -------------------------------------------------
        # Step 2: Cluster-level top merchants
        # -------------------------------------------------
        cluster_top_merchants = (
            user_merchant
            .merge(
                customer_clusters[["User_Id", "cluster_id"]],
                on="User_Id",
                how="left"
            )
            .groupby(["cluster_id", "Mer_Id"], as_index=False)
            .agg(Points=("Points", "sum"))
        )

        cluster_top_merchants = (
            cluster_top_merchants
            .sort_values(["cluster_id", "Points"], ascending=[True, False])
            .groupby("cluster_id")
            .head(10)
        )

        cluster_dict = {
            cid: group["Mer_Id"].tolist()
            for cid, group in cluster_top_merchants.groupby("cluster_id")
        }

        # -------------------------------------------------
        # Step 3: User-level recent merchants
        # -------------------------------------------------
        user_top_merchants = (
            user_merchant[user_merchant.Trx_Age > self.min_recency_days]
            .sort_values(["User_Id", "Points"], ascending=[True, False])
            .groupby("User_Id")
            .head(3)
            .groupby("User_Id")["Mer_Id"]
            .apply(list)
            .reset_index()
        )

        user_dict = dict(zip(user_top_merchants.User_Id, user_top_merchants.Mer_Id))

        # -------------------------------------------------
        # Step 4: Merge user + cluster recommendations
        # -------------------------------------------------
        recommendations = []

        for _, row in customer_clusters.iterrows():
            uid = row.User_Id
            cluster_id = row.cluster_id

            cluster_list = cluster_dict.get(cluster_id, [])
            user_list = user_dict.get(uid, [])

            # Prioritize user behavior, then cluster popularity
            final_list = user_list + [m for m in cluster_list if m not in user_list]
            final_list = final_list[:3] if final_list else []

            # -------------------------------
            # Explainability
            # -------------------------------
            explained_recs = []
            for m in final_list:
                if m in user_list:
                    reason = "Based on your recent activity"
                else:
                    reason = "Popular among users in the same cluster"

                explained_recs.append({
                    "merchant_id": m,
                    "reason": reason
                })

            recommendations.append({
                "User_Id": uid,
                "cluster_id": int(cluster_id) if pd.notna(cluster_id) else None,
                "recommendation_type": "Hybrid (User + Cluster)",
                "top_merchants": final_list,
                "recommendations": explained_recs
            })

        rec_df = pd.DataFrame(recommendations)

        # -------------------------------------------------
        # Step 5: Auto-save recommendations
        # -------------------------------------------------
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        rec_df.to_parquet(self.save_path, index=False)
        print(f"✅ [Recommendation] Recommendations saved to {self.save_path}")

        return rec_df
        print("✅ [Recommendation] Recommendation generation completed.")  