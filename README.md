# End-to-End Customer Merchant Preference Prediction System for WE Telecom

## Executive Summary

This is a **comprehensive data science project** that predicts customer merchant preferences using advanced clustering and recommendation techniques. The system leverages **Fuzzy C-Means (FCM) clustering**, machine learning feature engineering, and intelligent recommendation algorithms to segment customers and provide personalized merchant recommendations. The project is production-ready with a Streamlit web interface for interactive exploration and real-time predictions.

---

## Project Overview

### Business Objective
Develop an intelligent system to segment WE Telecom customers into distinct behavioral clusters and recommend suitable merchants based on historical transaction patterns and customer characteristics.

### Technical Approach
- **Data Pipeline**: ETL process for transaction data normalization and feature engineering
- **Feature Engineering**: Comprehensive customer behavioral metrics extraction
- **Clustering Algorithm**: Fuzzy C-Means (FCM) for soft clustering with membership probabilities
- **Recommendation Engine**: Content-based and collaborative filtering hybrid approach
- **Production Interface**: Interactive Streamlit application for real-time recommendations

### Key Technologies
- Python 3.x
- Scikit-learn, Pandas, NumPy
- Scikit-Fuzzy for Fuzzy C-Means implementation
- Streamlit for web interface
- Parquet for efficient data storage

---

## Project Structure & File Documentation

🗂️ Project Structure

smart-merchant-recommendation-system/
│
├── app/
│   └── app.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── clustering.py
│   ├── recommendation.py
│   ├── inference.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── clean/
│
├── models/
│
├── assets/
│   └── cover.png
│
├── metadata/
│
├── requirements.txt
└── README.md

### 📁 Root Level Files

#### `README.md`
This comprehensive documentation file containing the complete project specification, architecture, usage instructions, and detailed file descriptions.

#### `requirements.txt`
Python package dependencies specification. Contains all required libraries and their versions for:
- Data manipulation and analysis
- Machine learning algorithms
- Clustering implementations
- Web application framework
- Utilities and helpers

To install dependencies:
```bash
pip install -r requirements.txt
```

---

### 📂 `app/` - Application Layer

#### `app/app.py`
**Purpose**: Production-grade Streamlit web application serving as the user interface for the entire system.

**Key Features**:
- Interactive dashboard for customer analysis
- Real-time merchant recommendations
- Random user generation for demo purposes
- Dynamic styling controls (color schemes, layouts)
- Session state management for smooth UX
- Sidebar controls for data exploration
- Caching mechanisms for performance optimization
- Integration with the `RecommendationService` from the inference module

**Main Components**:
- Page configuration and theming
- Recommendation service initialization
- Session state variables for stateful interactions
- Sidebar controls for user interaction
- Multiple display modes and visualizations
- Data filtering and exploration capabilities

**Usage**:
```bash
streamlit run app/app.py
```

---

### 📂 `src/` - Core Processing Modules

#### `src/config.py`
**Purpose**: Central configuration and utility module managing project paths, settings, and data format conversions.

**Key Responsibilities**:
- Project root path resolution using `pathlib.Path`
- Define directory constants:
  - `DATA_RAW`: Path to raw data folder
  - `DATA_CLEAN`: Path to cleaned data folder
  - `MODELS`: Path to trained model storage
- Global random state for reproducibility (`RANDOM_STATE = 42`)
- CSV to Parquet conversion function for optimized data handling
- Ensure transactions data exists in efficient Parquet format

**Functions**:
- `ensure_transactions_parquet()`: Converts CSV files to Parquet format for faster I/O operations

**Usage**: Imported by all modules requiring path constants and configuration settings.

---

#### `src/preprocessing.py`
**Purpose**: Data loading and validation module handling raw transaction data ingestion.

**Core Components**:
- `TransactionLoader` class: Main interface for loading transaction data
  - Validates required columns presence
  - Loads data from Parquet files
  - Provides data shape and integrity information
  - Required columns:
    - `User_Id`: Customer identifier
    - `Mer_Id`: Merchant identifier
    - `Trx_Vlu`: Transaction value
    - `Points`: Loyalty points
    - `Customer_Age`: Customer age
    - `Trx_Age`: Transaction recency
    - `Category In English`: Merchant category

**Key Methods**:
- `load()`: Load and validate transaction data
- Data integrity checks and logging
- Comprehensive progress reporting with visual indicators

**Logging Features**: Detailed logging of data loading progress with step indicators.

---

#### `src/features.py`
**Purpose**: Feature engineering module that transforms raw transactions into customer-level behavioral features.

**Responsibilities**:
- Aggregate transaction-level data to customer level
- Create meaningful behavioral metrics:
  - Total spending patterns
  - Transaction frequency
  - Merchant diversity (unique merchants visited)
  - Category preferences
  - Average transaction values
  - Loyalty points accumulation
  - Customer lifetime value indicators
  - Recency and frequency metrics

**Output**: Customer feature vectors suitable for clustering algorithms.

---

#### `src/clustering.py`
**Purpose**: Customer segmentation engine using advanced clustering techniques.

**Main Class**: `CustomerClustering`

**Clustering Pipeline**:
1. **Data Normalization**: StandardScaler for feature standardization
2. **Dimensionality Reduction**: PCA (90% variance retention) to reduce feature space while preserving information
3. **Hard Clustering**: KMeans for initial segmentation
4. **Soft Clustering**: Fuzzy C-Means for probabilistic cluster membership

**Features**:
- Configurable cluster count
- Automatic cluster naming based on behavioral characteristics
- Behavioral insights generation for each cluster
- Automatic model persistence to disk
- Cluster visualization and statistics

**Output Artifacts**:
- FCM centers (centroids in reduced PCA space)
- FCM membership matrix (soft assignments with probabilities)
- Customer cluster assignments
- Cluster enrichment data with behavioral insights

---

#### `src/train.py`
**Purpose**: Main orchestration module executing the complete machine learning pipeline.

**Pipeline Orchestration**:
1. Load and validate configuration
2. Perform train/test split (default 70% train, 30% test)
3. Load training transactions
4. Extract customer features
5. Execute customer clustering
6. Train recommendation engine
7. Save all models and artifacts
8. Generate evaluation metrics

**Main Function**: `run_pipeline(n_clusters: int = 4)`

**Process Flow**:
- Data loading and validation
- Feature engineering
- Clustering model training
- Merchant recommendation model training
- Artifact persistence
- Pipeline logging and progress tracking

---

#### `src/inference.py`
**Purpose**: Production inference service providing real-time predictions and recommendations.

**Core Class**: `RecommendationService`

**Key Capabilities**:
- Load pre-trained models and artifacts
- Predict cluster membership for new customers
- Generate personalized merchant recommendations
- Handle real-time inference requests
- Manage customer cluster assignments database

**Methods**:
- Initialize with trained models
- Predict customer cluster
- Recommend merchants based on cluster and behavioral profile
- Retrieve cluster information
- Score merchant suitability

---

#### `src/recommendation.py`
**Purpose**: Merchant recommendation engine combining clustering insights with transaction history.

**Main Class**: `MerchantRecommender`

**Recommendation Strategy**:
- Use customer cluster membership (soft assignment probabilities)
- Analyze merchant popularity within clusters
- Calculate merchant affinity scores
- Rank merchants by recommendation score
- Filter recommendations based on business rules

**Features**:
- Personalized recommendations per customer
- Confidence scores for each recommendation
- Merchant diversity in results
- Transaction-based ranking
- Category-based filtering options

---

#### `src/evaluation.py`
**Purpose**: Model performance evaluation and validation module.

**Evaluation Metrics**:
- Clustering quality metrics:
  - Silhouette coefficient
  - Davies-Bouldin index
  - Calinski-Harabasz score
- Recommendation quality metrics:
  - Precision@K
  - Recall@K
  - NDCG (Normalized Discounted Cumulative Gain)
  - Coverage
  - Recommendation diversity metrics

**Key Functions**:
- Evaluate clustering performance
- Validate recommendation quality
- Generate evaluation reports
- Visualize cluster quality

---

#### `src/split.py`
**Purpose**: Data splitting module ensuring proper train/test separation.

**Function**: `per_user_time_split(test_size: float = 0.3)`

**Splitting Strategy**:
- **Per-User Temporal Split**: Ensures each customer's data is split chronologically
- Prevents data leakage by respecting transaction timestamps
- Maintains customer-consistent splits across datasets
- Configurable test size ratio

**Outputs**:
- Training dataset: Historical transactions for model training
- Test dataset: Future transactions for model evaluation

---

#### `src/utils.py`
**Purpose**: Utility functions providing common operations across modules.

**Key Utilities**:
- `assert_columns_exist()`: Validate required columns in DataFrames
- `save_parquet()`: Efficient Parquet file writing
- `save_model()`: Serialize and persist trained models
- `setup_logger()`: Configure logging infrastructure
- Data validation helpers
- File I/O operations
- Path utilities

**Usage**: Imported by all modules requiring common utility functions.

---

#### `src/__init__.py`
**Purpose**: Python package initialization file enabling module imports.

---

### 📂 `Data/` - Data Storage

#### `Data/raw/`
Contains original, unprocessed transaction data:
- `Cleaned_Data_Merchant_Level_2.csv`: Raw transaction dataset (converted to Parquet for efficiency)
- `transactions.parquet`: Optimized Parquet format of transaction data

**Data Fields**:
- Customer identifiers
- Merchant identifiers
- Transaction values
- Loyalty points
- Customer demographics
- Transaction metadata
- Merchant categories

#### `Data/clean/`
Contains processed and engineered datasets:
- Train/test split data
- Feature-engineered customer datasets
- Aggregated customer metrics

---

### 📂 `models/` - Trained Model Artifacts

#### `models/fcm_centers.npy`
**Purpose**: Trained Fuzzy C-Means cluster centers stored as NumPy array.

**Content**: 
- K × D matrix where K = number of clusters, D = feature dimensions
- Represents optimal cluster centroids in PCA-reduced feature space
- Used for new customer cluster assignment during inference

#### `models/fcm_membership.npy`
**Purpose**: FCM membership matrix storing soft cluster assignments.

**Content**:
- N × K matrix where N = number of training customers, K = number of clusters
- Contains membership probabilities for each customer-cluster pair
- Values range [0, 1], row-wise sum = 1
- Enables customer-cluster affinity analysis

**Additional Saved Models**:
- PCA transformer for feature space reduction
- KMeans model for reference
- Scaler artifacts for inference
- Recommendation model parameters

---

### 📂 `metadata/` - Project Metadata

Reserved directory for storing:
- Data dictionaries
- Feature descriptions
- Cluster characteristics summaries
- Model performance metrics
- Data quality reports

---

### 📂 `assets/` - Static Resources

Directory for storing:
- Project images and visualizations
- Documentation graphics
- Icon files
- Static content for web interface
- Data quality reports

---

### 📂 `noteBook/` - Exploratory Data Analysis

#### `End-to-End Data Science Project Predicting Customer Merchant Preferences WE Telecom.ipynb`
**Purpose**: Comprehensive Jupyter notebook documenting exploratory data analysis and development process.

**Contents**:
- Data loading and initial exploration
- Descriptive statistics and distributions
- Data quality assessments
- Feature engineering demonstrations
- Clustering algorithm experimentation
- Recommendation logic validation
- Model performance visualization
- Business insights derivation

**Key Sections**:
- Data overview and profiling
- Missing values analysis
- Outlier detection
- Feature correlation analysis
- Clustering visualization
- Recommendation results showcase
- Business recommendations and conclusions

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager
- Git (optional, for version control)

### Step 1: Clone/Download Project
```bash
git clone <repository-url>
cd project_root
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Ensure Data Exists
- Place `Cleaned_Data_Merchant_Level_2.csv` in `Data/raw/` directory
- Or parquet file if already converted

---

## Usage Guide

### Training Pipeline

To train the complete model:

```bash
python -c "from src.train import run_pipeline; run_pipeline(n_clusters=4)"
```

This executes:
1. Data loading and validation
2. Feature engineering
3. Customer clustering (FCM with K=4)
4. Recommendation model training
5. Model serialization

### Running the Web Application

Launch the interactive Streamlit interface:

```bash
streamlit run app/app.py
```

Then navigate to `http://localhost:8501` in your browser.

**Features**:
- Explore customer clusters
- Generate recommendations for any customer
- View cluster characteristics
- Test with random users
- Customize UI styling

### Batch Inference

```python
from src.inference import RecommendationService

# Initialize service with trained models
service = RecommendationService()

# Get recommendations for a customer
user_id = 12345
recommendations = service.get_recommendations(user_id, top_k=10)
print(recommendations)
```

---

## Project Workflow

```
Raw Transaction Data (CSV)
         ↓
   Preprocessing
         ↓
  Feature Engineering
         ↓
   Clustering (FCM)
  ↙              ↘
Customer         Cluster
Segments      Characteristics
         ↓
 Recommendation
   Engine
         ↓
Real-time Predictions
  (Streamlit App)
```

---

## Model Architecture

### Clustering Pipeline
1. **Standardization**: StandardScaler normalizes features to zero mean, unit variance
2. **Dimensionality Reduction**: PCA reduces to 90% variance, eliminating noise
3. **Segmentation**: 
   - KMeans for hard clustering (reference)
   - Fuzzy C-Means for soft clustering (primary)
4. **Enrichment**: Behavioral insights extraction per cluster

### Recommendation Pipeline
1. **Customer Cluster Assignment**: Soft membership using FCM
2. **Merchant Popularity**: Calculate within-cluster merchant metrics
3. **Scoring**: Combine cluster affinity with merchant popularity
4. **Ranking**: Sort by recommendation score
5. **Filtering**: Apply business rules and diversification

---

## Performance & Optimization

### Data Optimization
- CSV → Parquet format conversion for 10-100x faster I/O
- In-memory caching with Streamlit `@st.cache_resource`
- Vectorized NumPy operations for computational efficiency

### Model Optimization
- PCA dimensionality reduction reducing compute cost
- FCM soft clustering reducing overfitting
- KMeans initialization for clustering stability

### Scalability Considerations
- Parquet format supports efficient column-based reads
- Can handle millions of transactions with distributed computing
- Inference service optimized for batch predictions

---

## Key Insights & Business Value

### Customer Segmentation
Identify distinct customer behavioral groups enabling targeted marketing strategies.

### Personalized Recommendations
Recommend merchants aligned with customer preferences and spending patterns.

### Merchant Growth
Help merchants understand and target their ideal customer segments.

### Campaign Optimization
Use clusters for targeted promotional campaigns and loyalty programs.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | Pandas, NumPy |
| Feature Engineering | Scikit-learn |
| Clustering | Scikit-Fuzzy, Scikit-learn |
| Visualization | Streamlit |
| Model Persistence | NumPy, Joblib |
| Data Format | Parquet, CSV |
| Language | Python 3.x |

---

## Future Enhancements

1. **Real-time Streaming**: Apache Kafka for transaction streaming
2. **Deep Learning**: Neural network-based embeddings for customers/merchants
3. **Explainability**: SHAP values for recommendation transparency
4. **A/B Testing**: Experimentation framework for recommendation strategies
5. **Model Versioning**: MLflow for model tracking and deployment
6. **API Deployment**: FastAPI REST service for production deployment
7. **Temporal Analysis**: Time-series modeling for seasonal patterns
8. **Feedback Loop**: User feedback integration for continuous improvement

---

## Troubleshooting

### Issue: "FileNotFoundError: CSV file not found"
**Solution**: Ensure `Cleaned_Data_Merchant_Level_2.csv` exists in `Data/raw/` directory.

### Issue: Streamlit app not launching
**Solution**: 
```bash
pip install streamlit --upgrade
streamlit run app/app.py --logger.level=debug
```

### Issue: Out of memory with large datasets
**Solution**: Use Parquet format for efficient data loading, consider sampling for exploration.

---

## Contributing Guidelines

1. Create a new branch for features/fixes
2. Maintain code documentation
3. Follow PEP 8 style guidelines
4. Include unit tests for new functionality
5. Submit pull requests with clear descriptions

---

## License

This project is provided for educational and business use within WE Telecom.

---
🧪 Model Outputs

The system provides:

Customer cluster assignment

FCM confidence score

Personalized merchant recommendations

Cluster-level fallback recommendations

Explainable recommendation reasons

🏗️ Engineering Highlights

Modular & scalable architecture

Clear separation between UI and business logic

Production-ready imports

Cloud-compatible (Streamlit Cloud)

Explainable ML decisions

--------------

## Contact & Support

👨‍💻 Developer

Abdallah Nabil Ragab
Data Scientist | Machine Learning Engineer | Software Engineer
M.Sc. in Business Information Systems

📧 Email: abdallah.nabil.ragab94@gmail.com

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Project Status**: Production Ready

