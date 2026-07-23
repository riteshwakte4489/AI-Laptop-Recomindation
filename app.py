"""
AI-Powered Laptop Market Intelligence Dashboard
------------------------------------------------
Streamlit app that clusters laptops into market segments using PCA + KMeans,
then lets the user explore those segments interactively.

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
    page_title="Laptop Market Intelligence",
    page_icon="💻",
    layout="wide",
)

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

    # Drop columns that carry no clustering signal
    drop_cols = [c for c in ["index", "is_touch_screen",
                              "secondary_storage_capacity",
                              "secondary_storage_type"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Numeric / categorical column split. "Model" is free text (near-unique
    # per row) so it is excluded from encoding — it's kept only for display.
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns.drop("Model")

    # Try loading pre-trained models if they exist and the cluster size matches
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
        except Exception as e:
            # Fall back to training online if loading fails
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
    # Re-run the same preprocessing for each k to report a comparison table.
    scores = []
    for k in k_range:
        _, sil = load_and_cluster(_df_path, k)
        scores.append({"k": k, "silhouette": round(sil, 3)})
    return pd.DataFrame(scores)


# -----------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------
st.sidebar.title("💻 Controls")

n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=8, value=2)

try:
    df, sil_score = load_and_cluster(DATA_PATH, n_clusters)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Place your laptops CSV in the same "
        "folder as app.py, or update DATA_PATH at the top of the script."
    )
    st.stop()

brands = sorted(df["brand"].unique())
selected_brands = st.sidebar.multiselect("Brand", brands, default=brands)

price_min, price_max = int(df["Price"].min()), int(df["Price"].max())
price_range = st.sidebar.slider("Price range (₹)", price_min, price_max, (price_min, price_max))

os_options = sorted(df["OS"].unique())
selected_os = st.sidebar.multiselect("Operating System", os_options, default=os_options)

filtered = df[
    df["brand"].isin(selected_brands)
    & df["OS"].isin(selected_os)
    & df["Price"].between(*price_range)
]

# -----------------------------------------------------------------------
# Header + KPIs
# -----------------------------------------------------------------------
st.title("🖥️ AI-Powered Laptop Market Intelligence Dashboard")
st.caption("Unsupervised market segmentation of laptops using PCA + KMeans clustering.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Laptops in view", f"{len(filtered):,}")
k2.metric("Avg. price", f"₹{filtered['Price'].mean():,.0f}" if len(filtered) else "—")
k3.metric("Clusters (k)", n_clusters)
k4.metric("Silhouette score", f"{sil_score:.3f}")

if sil_score < 0.25:
    st.info(
        "Note: silhouette scores below 0.25 indicate weak/overlapping cluster "
        "separation. Treat segments as directional patterns, not hard boundaries."
    )

st.divider()

# -----------------------------------------------------------------------
# PCA scatter plot
# -----------------------------------------------------------------------
st.subheader("Market segments (PCA projection)")
fig = px.scatter(
    filtered,
    x="PC1", y="PC2",
    color=filtered["Cluster"].astype(str),
    hover_data=["brand", "Model", "Price", "ram_memory", "processor_tier"],
    labels={"color": "Cluster"},
    title=None,
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------
# Cluster profiles
# -----------------------------------------------------------------------
st.subheader("Cluster profiles")

profile = (
    df.groupby("Cluster")
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
    fig2 = px.box(df, x=df["Cluster"].astype(str), y="Price", labels={"x": "Cluster"})
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Brand mix by cluster")
    brand_mix = df.groupby([df["Cluster"].astype(str), "brand"]).size().reset_index(name="count")
    fig3 = px.bar(brand_mix, x="Cluster", y="count", color="brand")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------
# k selection helper
# -----------------------------------------------------------------------
with st.expander("How was k chosen? (silhouette comparison)"):
    st.write(
        "Silhouette score measures how well-separated the clusters are "
        "(higher is better, max 1.0). Use this to sanity-check the slider above."
    )
    sweep_df = silhouette_sweep(DATA_PATH)
    st.dataframe(sweep_df, use_container_width=True)
    fig4 = px.line(sweep_df, x="k", y="silhouette", markers=True)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------
# Raw data table
# -----------------------------------------------------------------------
with st.expander("View filtered raw data"):
    st.dataframe(
        filtered[["brand", "Model", "Price", "Rating", "ram_memory",
                  "processor_tier", "OS", "Cluster"]],
        use_container_width=True,
    )
