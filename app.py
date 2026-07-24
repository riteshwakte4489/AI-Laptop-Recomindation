"""
AI-Powered Laptop Recommendation Engine
-----------------------------------------
Streamlit app that recommends laptops based on budget and needs.
Under the hood, laptops are also segmented into market clusters via
PCA + KMeans (tucked into an "Advanced" section for anyone curious).

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Laptop Recommendation Engine",
    page_icon="💻",
    layout="wide",
)

# -----------------------------------------------------------------------
# Theme — dark navy / purple / teal
# -----------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0d1220 0%, #131a2e 100%);
    }
    h1, h2, h3 { color: #e8ecf7 !important; }
    p, span, label, .stCaption { color: #b8c0d9 !important; }

    section[data-testid="stSidebar"] {
        background: #0a0e1a;
        border-right: 1px solid #2a3352;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a2140 0%, #201847 100%);
        border: 1px solid #3d3a6e;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label { color: #7fdbd4 !important; }
    div[data-testid="stMetric"] div { color: #ffffff !important; }

    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #6c4cd1 0%, #2fb8a8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2a3352;
        border-radius: 10px;
        overflow: hidden;
    }

    .hero-box {
        background: linear-gradient(135deg, #1a2140 0%, #241a4a 60%, #17263f 100%);
        border: 1px solid #3d3a6e;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .hero-box h1 { margin-bottom: 4px; }

    div[data-baseweb="select"] > div, .stSlider {
        color: #e8ecf7;
    }
</style>
""", unsafe_allow_html=True)

# Robust data path resolution
if os.path.exists("laptops.csv"):
    DATA_PATH = "laptops.csv"
elif os.path.exists("laptops (1).csv"):
    DATA_PATH = "laptops (1).csv"
else:
    DATA_PATH = "laptops.csv"


# -----------------------------------------------------------------------
# Data loading + clustering pipeline (cached so it only runs once)
# -----------------------------------------------------------------------
@st.cache_data
def load_and_cluster(path: str, n_clusters: int):
    df = pd.read_csv(path)

    drop_cols = [c for c in ["index", "is_touch_screen",
                              "secondary_storage_capacity",
                              "secondary_storage_type"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns.drop("Model")

    preprocessor_path = "preprocessor.joblib"
    pca_path = "pca.joblib"
    kmeans_path = "kmeans.joblib"

    loaded_successfully = False
    if os.path.exists(preprocessor_path) and os.path.exists(pca_path) and os.path.exists(kmeans_path) and n_clusters == 2:
        try:
            preprocessor = joblib.load(preprocessor_path)
            pca = joblib.load(pca_path)
            kmeans = joblib.load(kmeans_path)

            x_processed = preprocessor.transform(df)
            x_pca = pca.transform(x_processed)
            cluster = kmeans.predict(x_processed)
            sil = silhouette_score(x_processed, cluster)
            loaded_successfully = True
        except Exception:
            pass

    if not loaded_successfully:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])

        x_processed = preprocessor.fit_transform(df)

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_processed)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster = kmeans.fit_predict(x_processed)

        sil = silhouette_score(x_processed, cluster)

    df["Cluster"] = cluster
    df["PC1"] = x_pca[:, 0]
    df["PC2"] = x_pca[:, 1]

    return df, sil


@st.cache_data
def silhouette_sweep(_df_path, k_range=range(2, 8)):
    scores = []
    for k in k_range:
        _, sil = load_and_cluster(_df_path, k)
        scores.append({"k": k, "silhouette": round(sil, 3)})
    return pd.DataFrame(scores)


try:
    df, sil_score = load_and_cluster(DATA_PATH, 2)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Place your laptops CSV in the same "
        "folder as app.py, or update DATA_PATH at the top of the script."
    )
    st.stop()

price_min, price_max = int(df["Price"].min()), int(df["Price"].max())

# -----------------------------------------------------------------------
# Hero header
# -----------------------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <h1>💻 Laptop Recommendation Engine</h1>
    <p style="font-size:16px;">Tell it your budget and needs — it finds the best-rated matches from real laptop listings.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Recommendation engine (hero feature)
# -----------------------------------------------------------------------
rcol1, rcol2, rcol3 = st.columns(3)
with rcol1:
    rec_budget = st.slider("💰 Max budget (₹)", price_min, price_max, price_max // 2)
with rcol2:
    rec_min_ram = st.selectbox("🧠 Min RAM (GB)", sorted(df["ram_memory"].unique()))
with rcol3:
    rec_use_case = st.selectbox(
        "🎯 Primary use", ["Any", "Gaming", "Office / Productivity", "Content creation"]
    )

use_case_masks = {
    "Gaming": df["gpu_type"].astype(str).str.contains("RTX|GTX|Radeon", case=False, na=False),
    "Office / Productivity": df["num_cores"] <= 8,
    "Content creation": df["ram_memory"] >= 16,
    "Any": pd.Series(True, index=df.index),
}

recommendations = (
    df[
        (df["Price"] <= rec_budget)
        & (df["ram_memory"] >= rec_min_ram)
        & use_case_masks[rec_use_case]
    ]
    .sort_values("Rating", ascending=False)
    .head(5)
)

st.markdown("&nbsp;", unsafe_allow_html=True)

if len(recommendations):
    st.markdown(f"#### ✨ Top picks within ₹{rec_budget:,} for **{rec_use_case}**")

    for _, row in recommendations.iterrows():
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 1.3, 1, 1])
            c1.markdown(f"**{row['brand']} — {row['Model']}**")
            c2.markdown(f"₹{row['Price']:,.0f}")
            c3.markdown(f"⭐ {row['Rating']}")
            c4.markdown(f"{int(row['ram_memory'])} GB RAM · {row['processor_tier']}")
            st.markdown("<hr style='margin:6px 0; border-color:#2a3352;'>", unsafe_allow_html=True)
else:
    st.warning("No laptops match those filters — try raising your budget or lowering the RAM requirement.")

st.divider()

# -----------------------------------------------------------------------
# Advanced: market segmentation analysis (collapsed by default)
# -----------------------------------------------------------------------
with st.expander("📊 Advanced: market segmentation analysis (PCA + KMeans)"):
    st.caption(
        "Under the hood, all laptops are also grouped into market clusters "
        "using PCA for dimensionality reduction and KMeans for clustering."
    )

    n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=3, key="adv_k")
    df_k, sil_k = load_and_cluster(DATA_PATH, n_clusters)

    m1, m2 = st.columns(2)
    m1.metric("Clusters (k)", n_clusters)
    m2.metric("Silhouette score", f"{sil_k:.3f}")

    if sil_k < 0.25:
        st.info(
            "Silhouette scores below 0.25 indicate weak/overlapping cluster "
            "separation — treat segments as directional patterns, not hard boundaries."
        )

    st.subheader("Market segments (PCA projection)")
    fig = px.scatter(
        df_k, x="PC1", y="PC2",
        color=df_k["Cluster"].astype(str),
        hover_data=["brand", "Model", "Price", "ram_memory", "processor_tier"],
        labels={"color": "Cluster"},
        color_discrete_sequence=["#7fdbd4", "#a78bfa", "#f0a868", "#e879a6",
                                  "#60a5fa", "#4ade80", "#fbbf24", "#f87171"],
    )
    fig.update_layout(
        height=450,
        plot_bgcolor="#131a2e", paper_bgcolor="#131a2e",
        font_color="#b8c0d9",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster profiles")
    profile = (
        df_k.groupby("Cluster")
        .agg(
            avg_price=("Price", "mean"),
            avg_rating=("Rating", "mean"),
            avg_ram=("ram_memory", "mean"),
            avg_cores=("num_cores", "mean"),
            top_brand=("brand", lambda x: x.mode()[0]),
            top_os=("OS", lambda x: x.mode()[0]),
            count=("Price", "size"),
        )
        .round(1)
        .reset_index()
    )
    st.dataframe(profile, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price distribution by cluster")
        fig2 = px.box(df_k, x=df_k["Cluster"].astype(str), y="Price", labels={"x": "Cluster"},
                       color=df_k["Cluster"].astype(str),
                       color_discrete_sequence=["#7fdbd4", "#a78bfa", "#f0a868", "#e879a6",
                                                 "#60a5fa", "#4ade80", "#fbbf24", "#f87171"])
        fig2.update_layout(plot_bgcolor="#131a2e", paper_bgcolor="#131a2e", font_color="#b8c0d9",
                            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Brand mix by cluster")
        brand_mix = df_k.groupby([df_k["Cluster"].astype(str), "brand"]).size().reset_index(name="count")
        fig3 = px.bar(brand_mix, x="Cluster", y="count", color="brand")
        fig3.update_layout(plot_bgcolor="#131a2e", paper_bgcolor="#131a2e", font_color="#b8c0d9")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("How was k chosen? (silhouette comparison)")
    sweep_df = silhouette_sweep(DATA_PATH)
    st.dataframe(sweep_df, use_container_width=True)
    fig4 = px.line(sweep_df, x="k", y="silhouette", markers=True)
    fig4.update_layout(plot_bgcolor="#131a2e", paper_bgcolor="#131a2e", font_color="#b8c0d9")
    fig4.update_traces(line_color="#7fdbd4", marker_color="#a78bfa")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Browse all laptops")
    st.dataframe(
        df_k[["brand", "Model", "Price", "Rating", "ram_memory",
              "processor_tier", "OS", "Cluster"]],
        use_container_width=True,
    )
