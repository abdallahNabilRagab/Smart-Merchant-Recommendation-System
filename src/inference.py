# src/inference.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_CLEAN, MODELS


class RecommendationService:
    """
    Lightweight inference layer for serving recommendations.
    Loads all relevant cluster & recommendation data.
    """

    def __init__(self):
        # -------------------------------------------------
        # Load recommendations
        # -------------------------------------------------
        rec_path = DATA_CLEAN / "recommendations.parquet"
        self.recommendations_df = pd.read_parquet(rec_path)
        print(f"🔹 [Inference] Recommendations loaded: {len(self.recommendations_df)}")

        # -------------------------------------------------
        # Load customer clusters
        # -------------------------------------------------
        clusters_path = DATA_CLEAN / "customer_clusters.parquet"
        self.customer_clusters_df = pd.read_parquet(clusters_path)
        print(f"🔹 [Inference] Customer clusters loaded: {len(self.customer_clusters_df)}")

        # -------------------------------------------------
        # Load cluster profiles
        # -------------------------------------------------
        profiles_path = DATA_CLEAN / "cluster_profiles.parquet"
        self.cluster_profiles_df = pd.read_parquet(profiles_path)
        print(f"🔹 [Inference] Cluster profiles loaded: {len(self.cluster_profiles_df)}")

        # -------------------------------------------------
        # Load FCM membership & centers
        # -------------------------------------------------
        self.fcm_membership = np.load(MODELS / "fcm_membership.npy")
        self.fcm_centers = np.load(MODELS / "fcm_centers.npy")
        print("🔹 [Inference] FCM membership and centers loaded.")

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def get(self, user_id: int):
        """
        Fetch recommendation record and cluster info for a given user.

        Returns
        -------
        dict or None
        """

        # -------------------------------
        # Fetch cluster info (MANDATORY)
        # -------------------------------
        cluster_row = self.customer_clusters_df[
            self.customer_clusters_df["User_Id"] == user_id
        ]

        if cluster_row.empty:
            print(f"❌ [Inference] No cluster found for User_Id={user_id}")
            return None

        cluster_data = cluster_row.iloc[0].to_dict()

        # -------------------------------
        # Fetch recommendations (OPTIONAL)
        # -------------------------------
        rec_row = self.recommendations_df[
            self.recommendations_df["User_Id"] == user_id
        ]

        if rec_row.empty:
            rec_data = {
                "recommendation_type": "Cluster-based fallback",
                "top_merchants": [],
                "recommendations": []
            }
            print(f"⚠️ [Inference] No personalized recommendations for User_Id={user_id} → fallback used")
        else:
            rec_data = rec_row.iloc[0].to_dict()

        # -------------------------------
        # Cluster profile & insights
        # -------------------------------
        cluster_id = cluster_data.get("cluster_id")

        profile_row = self.cluster_profiles_df[
            self.cluster_profiles_df["cluster_id"] == cluster_id
        ]

        cluster_insights = (
            profile_row["cluster_insights"].values[0]
            if not profile_row.empty
            else ""
        )

        # -------------------------------
        # Final result
        # -------------------------------
        result = {
            "User_Id": user_id,
            "cluster_id": cluster_id,
            "cluster_name": cluster_data.get("cluster_name", "N/A"),
            "cluster_insights": cluster_insights,
            "recommendation_type": rec_data.get("recommendation_type", "Unknown"),
            "top_merchants": rec_data.get("top_merchants", []),
            "recommendations": rec_data.get("recommendations", [])
        }

        return result
