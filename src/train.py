from src.config import TRANSACTIONS_TRAIN_PATH, DATA_CLEAN, MODELS
from src.preprocessing import TransactionLoader
from src.features import CustomerFeatureBuilder
from src.clustering import CustomerClustering
from src.recommendation import MerchantRecommender
from src.utils import save_parquet, save_model, setup_logger

from src.split import per_user_time_split


def run_pipeline(n_clusters: int = 4):

    setup_logger()

    print("🔹 [Pipeline] Starting pipeline...")

    # -------------------------------------------------
    # Train / Test Split
    # -------------------------------------------------
    print("🔸 [Pipeline] Ensuring train/test split exists...")
    per_user_time_split(test_size=0.3)

    # -------------------------------------------------
    # Load TRAIN data
    # -------------------------------------------------
    print("🔸 [Pipeline] Loading transactions TRAIN data...")
    transactions = TransactionLoader(
        TRANSACTIONS_TRAIN_PATH
    ).load()

    print(f"✅ [Pipeline] Loaded transactions: {transactions.shape[0]} rows, {transactions.shape[1]} cols")

    # -------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------
    print("🔸 [Pipeline] Building customer features...")
    features = CustomerFeatureBuilder().build(transactions)

    print(f"✅ [Pipeline] Features built: {features.shape[0]} customers, {features.shape[1]} features")

    print("🔸 [Pipeline] Saving customer_features.parquet...")
    save_parquet(features, DATA_CLEAN / "customer_features.parquet")
    print("✅ [Pipeline] customer_features.parquet saved")

    # -------------------------------------------------
    # Clustering
    # -------------------------------------------------
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

    print(f"🔸 [Pipeline] Clustering using columns: {clustering_cols}")
    print("🔸 [Pipeline] Running clustering...")

    clustering = CustomerClustering(n_clusters=n_clusters)

    clusters_df, membership = clustering.fit(
        X=X,
        features_df=features
    )

    print("✅ [Pipeline] Clustering completed")

    # -------------------------------------------------
    # Merge cluster outputs with features
    # -------------------------------------------------
    features = features.join(
        clusters_df.set_index("User_Id"),
        on="User_Id"
    )

    print("📌 [Pipeline] Cluster information merged into features")

    # -------------------------------------------------
    # Save Models
    # -------------------------------------------------
    print("🔸 [Pipeline] Saving models...")
    save_model(clustering.scaler, MODELS / "scaler.pkl")
    save_model(clustering.pca, MODELS / "pca.pkl")
    save_model(clustering.kmeans, MODELS / "kmeans.pkl")
    save_model(clustering.fcm_membership_, MODELS / "fcm_membership.pkl")

    print("✅ [Pipeline] Models saved successfully")

    # -------------------------------------------------
    # Recommendation
    # -------------------------------------------------
    print("🔸 [Pipeline] Generating recommendations...")
    recommendations = MerchantRecommender().recommend(
        customer_clusters=features,
        transactions=transactions
    )

    print(f"✅ [Pipeline] Recommendations generated: {recommendations.shape[0]} rows")

    print("🔸 [Pipeline] Saving recommendations.parquet...")
    save_parquet(
        recommendations,
        DATA_CLEAN / "recommendations.parquet"
    )

    print("✅ [Pipeline] recommendations.parquet saved")
    print("🎉 Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
