import numpy as np
import pandas as pd
import skfuzzy as fuzz
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.config import RANDOM_STATE, DATA_CLEAN, MODELS

class CustomerClustering:
    """
    Customer segmentation using:
    - StandardScaler
    - PCA (90% variance)
    - KMeans (hard clustering)
    - Fuzzy C-Means (soft clustering)

    Enrichment:
    - Cluster names
    - Behavioral insights
    - Automatic saving of all outputs
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.90, random_state=RANDOM_STATE)

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=20
        )

        self.fcm_centers_ = None
        self.fcm_membership_ = None
        self.cluster_profiles_ = None

    # -------------------------------------------------
    # Main Fit
    # -------------------------------------------------
    def fit(self, X: pd.DataFrame, features_df: pd.DataFrame | None = None):
        """
        Fit clustering models and enrich clusters.
        Automatically saves all outputs to disk.

        Returns
        -------
        clusters_df : pd.DataFrame
            User_Id, cluster_id, cluster_name, cluster_insights

        membership_matrix : np.ndarray
            Fuzzy membership matrix
        """

        print("🔹 [Clustering] Starting clustering pipeline...")

        # Scaling
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # KMeans
        hard_labels = self.kmeans.fit_predict(X_pca)

        # Fuzzy C-Means
        centers, membership, *_ = fuzz.cluster.cmeans(
            X_scaled.T,
            c=self.n_clusters,
            m=2.0,
            error=0.005,
            maxiter=300,
            seed=RANDOM_STATE
        )

        self.fcm_centers_ = centers
        self.fcm_membership_ = membership.T  # samples x clusters

        # -------------------------------------------------
        # Enrichment
        # -------------------------------------------------
        print("🔸 [Clustering] Generating cluster insights...")

        cluster_names = self._generate_cluster_names(hard_labels)

        clusters_df = pd.DataFrame({
            "User_Id": X.index,
            "cluster_id": hard_labels,
            "cluster_name": [cluster_names[c] for c in hard_labels],
            "dominant_membership_score": self.fcm_membership_.max(axis=1)
        })

        if features_df is not None:
            profiles = self._build_cluster_profiles(clusters_df, features_df)
            clusters_df = clusters_df.merge(
                profiles,
                on="cluster_id",
                how="left"
            )
            # Save cluster profiles
            profiles_path = DATA_CLEAN / "cluster_profiles.parquet"
            profiles.to_parquet(profiles_path, index=False)
            print(f"✅ Cluster profiles saved: {profiles_path}")

        # -------------------------------------------------
        # Save all outputs
        # -------------------------------------------------
        clusters_path = DATA_CLEAN / "customer_clusters.parquet"
        clusters_df.to_parquet(clusters_path, index=False)
        print(f"✅ Customer clusters saved: {clusters_path}")

        fcm_membership_path = MODELS / "fcm_membership.npy"
        np.save(fcm_membership_path, self.fcm_membership_)
        print(f"✅ FCM membership saved: {fcm_membership_path}")

        fcm_centers_path = MODELS / "fcm_centers.npy"
        np.save(fcm_centers_path, self.fcm_centers_)
        print(f"✅ FCM centers saved: {fcm_centers_path}")

        print("✅ [Clustering] Cluster enrichment completed.")
        print("🔹 [Clustering] Pipeline finished successfully.\n")

        return clusters_df, self.fcm_membership_

    # -------------------------------------------------
    # Helper Methods
    # -------------------------------------------------
    def _generate_cluster_names(self, labels: np.ndarray) -> dict:
        default_names = [
            "Loyal Customers",
            "High Value Customers",
            "Occasional Buyers",
            "Price Sensitive Users",
            "New / Low Activity Users"
        ]

        return {
            c: default_names[i % len(default_names)]
            for i, c in enumerate(np.unique(labels))
        }

    def _build_cluster_profiles(
        self,
        clusters_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:

        merged = clusters_df.merge(features_df, on="User_Id", how="left")

        profiles = (
            merged
            .groupby("cluster_id")
            .agg(
                avg_transactions=("transaction_count", "mean"),
                avg_spend=("average_transaction_value", "mean"),
                avg_unique_categories=("unique_categories", "mean")
            )
            .reset_index()
        )

        profiles["cluster_insights"] = profiles.apply(
            self._generate_insight_text,
            axis=1
        )

        return profiles[["cluster_id", "cluster_insights"]]

    def _generate_insight_text(self, row) -> str:
        insights = []

        insights.append(
            "High frequency users"
            if row.avg_transactions > 20
            else "Low to medium activity users"
        )

        insights.append(
            "High spending behavior"
            if row.avg_spend > 200
            else "Medium spending behavior"
        )

        if row.avg_unique_categories > 3:
            insights.append("Diverse category interests")

        return " • ".join(insights)
