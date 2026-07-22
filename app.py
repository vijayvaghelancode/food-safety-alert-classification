"""
Food-Safety Alert Classification — Streamlit Dashboard
--------------------------------------------------------
Loads the model pipeline trained in Food_Safety_Alert_Classification.ipynb
(outputs/model_pipeline.joblib) and provides:
  1) An interactive predictor: product + origin + destination + month -> hazard category
  2) A model comparison view
  3) An EDA dashboard over historical RASFF-derived notifications

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub, then deploy on https://share.streamlit.io
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Food-Safety Alert Classifier", page_icon="🥫", layout="wide")

OUTPUTS = "outputs"


@st.cache_resource
def load_model():
    pipe = joblib.load(f"{OUTPUTS}/model_pipeline.joblib")
    le = joblib.load(f"{OUTPUTS}/label_encoder.joblib")
    with open(f"{OUTPUTS}/model_metadata.json") as f:
        meta = json.load(f)
    return pipe, le, meta


@st.cache_data
def load_data():
    df = pd.read_csv(f"{OUTPUTS}/clean_data_sample.csv")
    comparison = pd.read_csv(f"{OUTPUTS}/model_comparison_baseline.csv") \
        if _exists(f"{OUTPUTS}/model_comparison_baseline.csv") else None
    tuned = pd.read_csv(f"{OUTPUTS}/model_comparison_tuned.csv") \
        if _exists(f"{OUTPUTS}/model_comparison_tuned.csv") else None
    return df, comparison, tuned


def _exists(path):
    import os
    return os.path.exists(path)


st.title("🥫 Food-Safety Alert Classification Dashboard")
st.caption(
    "Predicts the likely **hazard category** of a food/feed safety notification "
    "from the product, origin country, destination country and month — trained on "
    "real, RASFF-derived EU food-safety notifications."
)

try:
    pipe, le, meta = load_model()
    df, comparison_df, tuned_df = load_data()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "⚠️ Model artifacts not found in `outputs/`. Run the notebook "
        "`Food_Safety_Alert_Classification.ipynb` first (or download `outputs.zip` "
        "from Colab) and place the `outputs/` folder next to `app.py`."
    )

tab_predict, tab_models, tab_eda = st.tabs(["🔮 Predict", "📊 Model comparison", "📈 Data explorer"])

# --------------------------------------------------------------------------- #
with tab_predict:
    if not model_loaded:
        st.stop()

    st.subheader("Predict the hazard category of a notification")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        product = st.text_input("Product / food description", value="frozen shrimps")
    with col2:
        origin_options = sorted(df["origin_country"].dropna().unique().tolist())
        origin = st.selectbox("Origin country", origin_options,
                               index=origin_options.index("China") if "China" in origin_options else 0)
    with col3:
        dest_options = sorted(df["destination_country"].dropna().unique().tolist())
        destination = st.selectbox("Destination country", dest_options,
                                    index=dest_options.index("Germany") if "Germany" in dest_options else 0)
    with col4:
        month = st.slider("Month", 1, 12, 6)

    year = st.number_input("Year", min_value=2000, max_value=2035, value=2024, step=1)

    if st.button("Predict hazard category", type="primary"):
        input_row = pd.DataFrame([{
            "food": product.lower().strip(),
            "origin_country": origin,
            "destination_country": destination,
            "year": year,
            "month": month,
        }])

        pred = pipe.predict(input_row)[0]
        pred_label = le.classes_[pred] if isinstance(pred, (int, np.integer)) else pred

        st.success(f"### Predicted hazard category: **{pred_label}**")

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba(input_row)[0]
            classes = le.classes_ if len(le.classes_) == len(proba) else pipe.classes_
            proba_df = pd.DataFrame({"hazard_category": classes, "probability": proba}) \
                .sort_values("probability", ascending=False).head(8)
            fig = px.bar(proba_df, x="probability", y="hazard_category", orientation="h",
                         title="Top predicted-probability hazard categories")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "⚠️ This is a statistical prediction based on historical notification patterns, "
            "not a certainty of an actual hazard. Use it to prioritise inspection, not to replace it."
        )

    st.divider()
    st.markdown(f"**Final model:** `{meta['final_model']}` &nbsp;|&nbsp; "
                f"**Test accuracy:** {meta['test_accuracy']:.3f} &nbsp;|&nbsp; "
                f"**Test macro-F1:** {meta['test_macro_f1']:.3f}")

# --------------------------------------------------------------------------- #
with tab_models:
    if not model_loaded:
        st.stop()
    st.subheader("Baseline model comparison")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
        fig = px.bar(comparison_df.sort_values("macro_f1"), x="macro_f1", y="model",
                     orientation="h", title="Macro-F1 by model (baseline)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("model_comparison_baseline.csv not found in outputs/.")

    st.subheader("Tuned model results (top 2 models)")
    if tuned_df is not None:
        st.dataframe(tuned_df, use_container_width=True)
    else:
        st.info("model_comparison_tuned.csv not found in outputs/.")

# --------------------------------------------------------------------------- #
with tab_eda:
    if not model_loaded:
        st.stop()
    st.subheader("Historical notification explorer")

    c1, c2 = st.columns(2)
    with c1:
        cat_counts = df["hazard_category"].value_counts().reset_index()
        cat_counts.columns = ["hazard_category", "count"]
        fig = px.bar(cat_counts, x="count", y="hazard_category", orientation="h",
                     title="Notifications by hazard category")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_origin = df["origin_country"].value_counts().head(15).reset_index()
        top_origin.columns = ["origin_country", "count"]
        fig = px.bar(top_origin, x="count", y="origin_country", orientation="h",
                     title="Top 15 origin countries")
        st.plotly_chart(fig, use_container_width=True)

    by_year = df.groupby("year").size().reset_index(name="count")
    fig = px.line(by_year, x="year", y="count", markers=True, title="Notifications per year (sample)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Browse raw sample data")
    st.dataframe(df.sample(min(200, len(df))), use_container_width=True)
