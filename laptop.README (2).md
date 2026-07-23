# AI-Powered Laptop Market Intelligence Dashboard

An interactive Streamlit dashboard that segments ~1,000 laptops into market
clusters (e.g. budget, gaming, premium ultrabook) using PCA for
dimensionality reduction and KMeans for unsupervised clustering.

## What it does

- Loads a laptop specs + price dataset
- Preprocesses features (imputation, scaling, one-hot encoding) via a
  scikit-learn `ColumnTransformer`
- Reduces the feature space to 2D with PCA for visualization
- Clusters laptops with KMeans (k is adjustable from the sidebar)
- Reports a silhouette score so you can judge cluster quality, not just
  trust the chosen k blindly
- Lets you filter by brand, price range, and OS, and explore each
  cluster's average price/RAM/rating/top brand

## Project structure

```
.
├── app.py              # Streamlit app
├── requirements.txt    # Python dependencies
├── laptops.csv          # your dataset (not included — see below)
└── README.md
```

## Setup

1. **Clone / download this folder.**

2. **Add your dataset.** Place your laptop CSV in the same folder as
   `app.py` and name it `laptops.csv` (or edit the `DATA_PATH` variable
   at the top of `app.py` to point elsewhere). The app expects these
   columns (same as the source Kaggle-style dataset):
   `brand, Model, Price, Rating, processor_brand, processor_tier,
   num_cores, num_threads, ram_memory, primary_storage_type,
   primary_storage_capacity, gpu_brand, gpu_type, display_size,
   resolution_width, resolution_height, OS, year_of_warranty`
   (plus optionally `index`, `is_touch_screen`,
   `secondary_storage_capacity`, `secondary_storage_type` — these are
   dropped automatically if present).

3. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   It will open at `http://localhost:8501`.

## Notes on the modeling

- `Model` (the free-text laptop name) is intentionally excluded from
  clustering — it's near-unique per row and would dominate the feature
  space if one-hot encoded, drowning out real signal like specs/price.
- The silhouette score for this dataset is modest (roughly 0.15–0.31
  depending on k) — laptop specs/price don't separate into
  razor-sharp clusters in real-world data. Treat the segments as
  useful directional groupings, not ground truth categories.
- k=2 currently gives the best silhouette score of the tested range
  (2–8). The sidebar slider lets you compare other values live and see
  how the score changes — check the "How was k chosen?" expander in
  the app.

## Deploying it live (optional)

Push this repo to GitHub, then deploy for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):
1. Sign in with GitHub
2. "New app" → pick this repo → set main file to `app.py`
3. Deploy — you'll get a public URL to share (e.g. on your resume/LinkedIn)

Make sure `laptops.csv` is included in the repo (or loaded from a public
URL) since Streamlit Cloud won't have access to your local files.
