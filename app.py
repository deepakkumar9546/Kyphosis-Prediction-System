import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "kyphosis.csv")

FEATURES = ["Age", "Number", "Start"]
TARGET = "Kyphosis"

st.set_page_config(page_title="Random Forest — Kyphosis", page_icon="🌳", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_random_forest(df: pd.DataFrame, n_estimators: int, max_depth):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )
    return model, acc, cm, importances


data_df = load_data()

st.title("🌳 Will spinal surgery result in Kyphosis?")
st.caption("Random forest classifier trained on `kyphosis.csv`")

st.sidebar.header("Model settings")
n_estimators = st.sidebar.slider("Number of trees", min_value=10, max_value=500, value=100, step=10)
max_depth_choice = st.sidebar.selectbox("Max tree depth", ["None", 2, 3, 4, 5, 6, 8, 10], index=0)
max_depth = None if max_depth_choice == "None" else int(max_depth_choice)

model, acc, cm, importances = train_random_forest(data_df, n_estimators, max_depth)

tab_predict, tab_explore, tab_data = st.tabs(["🔮 Predict", "📊 Explore", "🗂️ Data"])

with tab_predict:
    col_form, col_metrics = st.columns([2, 1])

    with col_form:
        st.subheader("Enter patient details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(
                "Age (months)",
                min_value=0,
                value=int(data_df["Age"].mean()),
                step=1,
            )
        with c2:
            number = st.number_input(
                "Number (vertebrae involved)",
                min_value=0,
                value=int(data_df["Number"].mean()),
                step=1,
            )
        with c3:
            start = st.number_input(
                "Start (topmost vertebra operated)",
                min_value=0,
                value=int(data_df["Start"].mean()),
                step=1,
            )

        if st.button("Predict Kyphosis", type="primary"):
            input_df = pd.DataFrame([[age, number, start]], columns=FEATURES)
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            present_proba = proba[classes.index("present")]

            if prediction == "present":
                st.error(f"⚠️ Kyphosis likely present (probability: {present_proba:.1%})")
            else:
                st.success(f"✅ Kyphosis likely absent (probability: {1 - present_proba:.1%})")

    with col_metrics:
        st.subheader("Model performance")
        st.metric("Accuracy", f"{acc:.1%}")
        st.caption(f"Trained on {len(data_df):,} records")

        st.subheader("Confusion matrix")
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=list(model.classes_),
            y=list(model.classes_),
            color_continuous_scale="Greens",
        )
        st.plotly_chart(cm_fig, width="stretch")

        st.subheader("Feature importance")
        st.plotly_chart(
            px.bar(importances, orientation="h", labels={"index": "Feature", "value": "Importance"}),
            width="stretch",
        )

with tab_explore:
    st.subheader("Feature vs Kyphosis")
    feature_choice = st.selectbox("Choose a feature", FEATURES, key="kyphosis_feature")
    fig = px.box(
        data_df, x=TARGET, y=feature_choice, color=TARGET,
        title=f"{feature_choice} by {TARGET}",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Target distribution")
    st.plotly_chart(px.histogram(data_df, x=TARGET), width="stretch")

with tab_data:
    st.subheader("Raw dataset")
    st.dataframe(data_df, width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(data_df.describe(), width="stretch")
