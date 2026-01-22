# src/evaluation.py

import pandas as pd
import numpy as np
from math import log2
from src.config import TRANSACTIONS_TEST_PATH, MODELS
from src.preprocessing import TransactionLoader
from src.features import CustomerFeatureBuilder
from src.recommendation import MerchantRecommender
from src.utils import load_model


# =========================================================
# Clustering Evaluation (UNCHANGED – scientific metrics)
# =========================================================
def evaluate_clustering(features: pd.DataFrame):
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score
    )

    clustering_cols = [
        "recency_days",
        "transaction_count",
        "total_transaction_value",
        "average_transaction_value",
        "total_points_used",
        "unique_merchants",
        "unique_categories",
        "log_transaction_count",
        "log_total_transaction_value",
        "log_total_points_used"
    ]

    X = features[clustering_cols]
    labels = features["cluster_id"]

    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    print("🔹 [Evaluation] Clustering Metrics:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"  - Calinski-Harabasz Index: {calinski:.4f}")
    print(f"  - Davies-Bouldin Index: {davies:.4f}\n")

    return {
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies
    }


# =========================================================
# Recommendation Metrics (6 METRICS – PROFESSIONAL)
# =========================================================
def evaluate_recommendation(
    transactions: pd.DataFrame,
    recommendations: pd.DataFrame,
    k: int = 10
):
    # 🔍 Smart detection of recommendation column
    possible_cols = ["recommended_merchants", "top_merchants", "recommendations"]
    rec_col = next((c for c in possible_cols if c in recommendations.columns), None)

    if rec_col is None:
        raise ValueError(
            "❌ Recommendation column not found. Expected one of: "
            + ", ".join(possible_cols)
        )

    # Ground truth: actual merchants per user
    user_actual = (
        transactions.groupby("User_Id")["Mer_Id"]
        .apply(set)
        .to_dict()
    )

    precisions, recalls, f1s, ndcgs = [], [], [], []
    hits = 0
    all_recommended_items = set()

    evaluated_users = 0

    for _, row in recommendations.iterrows():
        user_id = row["User_Id"]
        recs = row[rec_col][:k]

        if user_id not in user_actual or len(recs) == 0:
            continue

        evaluated_users += 1
        actual = user_actual[user_id]
        recs_set = set(recs)

        all_recommended_items.update(recs_set)

        # ------------------
        # HitRate
        # ------------------
        hit = 1 if len(recs_set & actual) > 0 else 0
        hits += hit

        # ------------------
        # Precision & Recall
        # ------------------
        tp = len(recs_set & actual)
        precision = tp / k
        recall = tp / len(actual) if len(actual) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

        # ------------------
        # F1-score
        # ------------------
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0)

        # ------------------
        # NDCG
        # ------------------
        dcg = 0.0
        for i, item in enumerate(recs):
            if item in actual:
                dcg += 1 / log2(i + 2)

        ideal_hits = min(len(actual), k)
        idcg = sum(1 / log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    # ================================
    # Final Metrics
    # ================================
    precision_at_k = np.mean(precisions)
    recall_at_k = np.mean(recalls)
    f1_at_k = np.mean(f1s)
    hitrate_at_k = hits / evaluated_users if evaluated_users > 0 else 0
    ndcg_at_k = np.mean(ndcgs)

    # Coverage
    total_merchants = transactions["Mer_Id"].nunique()
    coverage = len(all_recommended_items) / total_merchants if total_merchants > 0 else 0

    print("🔹 [Evaluation] Recommendation Metrics:")
    print(f"  - Recall@{k}:     {recall_at_k:.4f}")
    print(f"  - Precision@{k}:  {precision_at_k:.4f}")
    print(f"  - F1-score@{k}:   {f1_at_k:.4f}")
    print(f"  - HitRate@{k}:    {hitrate_at_k:.4f}")
    print(f"  - NDCG@{k}:       {ndcg_at_k:.4f}")
    print(f"  - Coverage:       {coverage:.4f}\n")

    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "f1_at_k": f1_at_k,
        "hitrate_at_k": hitrate_at_k,
        "ndcg_at_k": ndcg_at_k,
        "coverage": coverage
    }


# =========================================================
# Full Evaluation Pipeline
# =========================================================
def run_evaluation():
    print("🔹 [Evaluation] Starting evaluation...")

    # 1️⃣ Load TEST data
    print("🔸 Loading TEST transactions...")
    test_transactions = TransactionLoader(TRANSACTIONS_TEST_PATH).load()

    # 2️⃣ Feature engineering (TEST only)
    print("🔸 Building customer features (TEST)...")
    test_features = CustomerFeatureBuilder().build(test_transactions)

    # 3️⃣ Load trained models
    print("🔸 Loading trained models...")
    scaler = load_model(MODELS / "scaler.pkl")
    pca = load_model(MODELS / "pca.pkl")
    kmeans = load_model(MODELS / "kmeans.pkl")

    # 4️⃣ Predict clusters
    print("🔸 Predicting clusters...")
    clustering_cols = [
        "recency_days",
        "transaction_count",
        "total_transaction_value",
        "average_transaction_value",
        "total_points_used",
        "unique_merchants",
        "unique_categories",
        "log_transaction_count",
        "log_total_transaction_value",
        "log_total_points_used"
    ]

    X_scaled = scaler.transform(test_features[clustering_cols])
    X_pca = pca.transform(X_scaled)
    test_features["cluster_id"] = kmeans.predict(X_pca)

    # 5️⃣ Evaluate clustering
    evaluate_clustering(test_features)

    # 6️⃣ Recommendations
    recommender = MerchantRecommender()
    print("🔸 Generating recommendations...")
    test_recommendations = recommender.recommend(
        test_features,
        test_transactions
    )

    # 7️⃣ Evaluate recommendations (6 metrics)
    evaluate_recommendation(
        test_transactions,
        test_recommendations,
        k=10
    )

    print("✅ [Evaluation] Completed successfully")


if __name__ == "__main__":
    run_evaluation()
