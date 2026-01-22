# app.py

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from inference import RecommendationService


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Merchant Recommendation",
    page_icon="🎯",
    layout="wide"
)

# -------------------------------------------------
# Load Service
# -------------------------------------------------
@st.cache_resource
def load_service():
    return RecommendationService()

service = load_service()

# -------------------------------------------------
# Initialize session state
# -------------------------------------------------
if "layout_mode" not in st.session_state:
    st.session_state.layout_mode = "wide"

if "cover_color" not in st.session_state:
    st.session_state.cover_color = "#ffffff"

if "text_color" not in st.session_state:
    st.session_state.text_color = "#000000"

if "random_users" not in st.session_state:
    st.session_state.random_users = []

# -------------------------------------------------
# Sidebar (Inputs + Style Controls + Instructions)
# -------------------------------------------------
with st.sidebar:
    st.header("🧭 Controls")

    # Random Users
    st.subheader("🎲 Try with Random Users")
    if st.button("Generate Random Users (5)"):
        all_user_ids = (
            service.customer_clusters_df["User_Id"]
            .dropna()
            .astype(int)
            .unique()
        )
        st.session_state.random_users = pd.Series(all_user_ids).sample(5).tolist()

    selected_user = None
    if st.session_state.random_users:
        selected_user = st.selectbox(
            "Select a User from random sample",
            st.session_state.random_users
        )

    # Manual Input
    st.subheader("✍️ Or Enter User ID Manually")
    manual_user = st.number_input(
        "Enter User ID",
        min_value=int(service.customer_clusters_df["User_Id"].min()),
        max_value=int(service.customer_clusters_df["User_Id"].max()),
        step=1
    )

    # Theme & Layout Controls
    st.markdown("---")
    st.subheader("🎨 Cover Style")

    st.session_state.cover_color = st.color_picker(
        "Cover Background Color",
        value=st.session_state.cover_color
    )

    st.session_state.text_color = st.color_picker(
        "Text Color",
        value=st.session_state.text_color
    )

    # Layout Toggle
    if st.button("Toggle Layout (Wide/Centered)"):
        st.session_state.layout_mode = "centered" if st.session_state.layout_mode == "wide" else "wide"
        st.query_params = {"layout": st.session_state.layout_mode}

    # Theme instructions (cannot be toggled programmatically)
    st.markdown("---")
    st.subheader("🌓 Theme (Official Streamlit)")
    st.markdown(
        """
        Streamlit Theme cannot be changed from inside the app.
        To switch Dark/Light mode:
        1. Click **Settings (⚙️)**  
        2. Go to **Theme**  
        3. Choose **Light** or **Dark**
        """
    )

    # Instructions
    st.markdown("---")
    st.subheader("📌 Instructions")
    st.markdown("""
    - Generate random users or enter a user ID manually.
    - Click **Get Recommendation** to view results.
    - Use the style controls to customize the cover.
    - Layout toggle changes the page width.
    """)

# -------------------------------------------------
# Apply Dynamic CSS based on user selection
# -------------------------------------------------
theme_css = f"""
<style>
    .cover-container {{
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        text-align: center;
        background: {st.session_state.cover_color};
    }}
    .cover-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 15px 0 5px 0;
        color: {st.session_state.text_color};
    }}
    .cover-subtitle {{
        font-size: 1.2rem;        /* تكبير العنوان الصغير */
        font-weight: 700;
        margin: 0 0 15px 0;
        color: #6c757d;           /* لون رصاصي */
        font-style: italic;       /* مائل */
    }}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# -------------------------------------------------
# Header (Title + Cover Image)
# -------------------------------------------------
st.markdown(
    """
    <div class="cover-container">
        <div class="cover-title">
            End-to-End Data Science Project Predicting Customer Merchant Preferences – WE Telecom
        </div>
        <div class="cover-subtitle">
            Smart Merchant Recommendation System
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Full width image
st.image("assets/cover.png", width=2000)

st.title("🎯 Smart Merchant Recommendation System")

# -------------------------------------------------
# Determine active user
# -------------------------------------------------
user_id = int(selected_user) if selected_user is not None else int(manual_user)

# -------------------------------------------------
# Recommendation Section
# -------------------------------------------------
if st.button("Get Recommendation"):

    result = service.get(user_id)

    if result is None:
        st.error("❌ No cluster information found for this user")
        st.stop()

    # -------------------------------
    # User Profile
    # -------------------------------
    st.subheader("👤 User Profile")
    col1, col2, col3 = st.columns(3)

    col1.metric("User ID", user_id)
    col2.metric("Cluster ID", result.get("cluster_id"))
    col3.metric("Cluster Name", result.get("cluster_name"))

    # -------------------------------
    # FCM Confidence Score
    # -------------------------------
    user_idx = service.customer_clusters_df.index[
        service.customer_clusters_df["User_Id"] == user_id
    ][0]

    confidence = float(
        service.fcm_membership[user_idx].max()
    )

    st.progress(confidence)
    st.caption(f"🔐 **Cluster Confidence Score (FCM):** {confidence:.2%}")

    # -------------------------------
    # Cluster Insights
    # -------------------------------
    cluster_insights = result.get("cluster_insights", "")
    if cluster_insights:
        st.subheader("📊 Cluster Insights")
        for insight in cluster_insights.split(" • "):
            st.write(f"• {insight}")

    # -------------------------------
    # Personalized Recommendations
    # -------------------------------
    st.subheader("⭐ Personalized Merchant Recommendations")

    explained_recs = result.get("recommendations", [])

    if explained_recs is None:
        explained_recs = []
    elif not isinstance(explained_recs, list):
        explained_recs = list(explained_recs)

    if len(explained_recs) == 0:
        st.warning(
            "⚠️ No personalized recommendations available. "
            "Using cluster-level fallback."
        )
    else:
        rec_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Merchant ID": r["merchant_id"],
                "Reason": r["reason"]
            }
            for i, r in enumerate(explained_recs)
        ])
        st.dataframe(rec_df, use_container_width=True)

    # -------------------------------
    # Cluster-Level Fallback Merchants
    # -------------------------------
    st.subheader("🏪 Top Merchants in This Cluster")

    cluster_id = result.get("cluster_id")

    cluster_recs = (
        service.recommendations_df[
            service.recommendations_df["cluster_id"] == cluster_id
        ]
        .explode("top_merchants")
        .groupby("top_merchants")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="Popularity")
    )

    if not cluster_recs.empty:
        cluster_recs.rename(
            columns={"top_merchants": "Merchant ID"},
            inplace=True
        )
        st.dataframe(cluster_recs, use_container_width=True)
    else:
        st.info("No cluster-level merchant data available.")

    # -------------------------------
    # Metadata
    # -------------------------------
    st.caption(
        f"Recommendation Type: **{result.get('recommendation_type')}**"
    )


# --------------------------------------------------------------
# Footer / Developer Info Section
# --------------------------------------------------------------
st.markdown("---")
st.subheader("👨‍💻 Developer Information")

st.markdown("""
**This application was fully developed and engineered by:**

### 🧑‍💻 **Abdallah Nabil Ragab**  
**Data Scientist | Machine Learning Engineer | Software Engineer**  
**M.Sc. in Business Information Systems**

If you have any suggestions, ideas, feature requests, or want to report issues,  
please feel free to send your feedback directly via email:

📩 **Email:** `abdallah.nabil.ragab94@gmail.com`  

I appreciate your thoughts and feedback that help improve this project.  
""")

st.markdown("---")
