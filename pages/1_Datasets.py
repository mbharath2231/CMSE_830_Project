import streamlit as st
import pandas as pd

st.title("📊 Datasets Overview & Summary")

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

tab1, tab2 = st.tabs(["📘 Matches Dataset", "📗 Deliveries Dataset"])

# -----------------------------
# TAB 1 — MATCHES
# -----------------------------
with tab1:
    st.subheader("🔍 Raw Data Preview")
    with st.expander("Show Matches Dataframe"):
        st.dataframe(matches)

    st.subheader("📐 Shape")
    st.write(f"Rows: **{matches.shape[0]}**, Columns: **{matches.shape[1]}**")

    st.subheader("🔤 Column Data Types")
    st.dataframe(matches.dtypes.rename("dtype"))

    st.subheader("📊 Missing Value Summary")
    st.dataframe(matches.isnull().sum().rename("missing_values"))

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(matches.describe(include="all"))

# -----------------------------
# TAB 2 — DELIVERIES
# -----------------------------
with tab2:
    st.subheader("🔍 Raw Data Preview")
    with st.expander("Show Deliveries Dataframe"):
        st.dataframe(deliveries)

    st.subheader("📐 Shape")
    st.write(f"Rows: **{deliveries.shape[0]}**, Columns: **{deliveries.shape[1]}**")

    st.subheader("🔤 Column Data Types")
    st.dataframe(deliveries.dtypes.rename("dtype"))

    st.subheader("📊 Missing Value Summary")
    st.dataframe(deliveries.isnull().sum().rename("missing_values"))

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(deliveries.describe(include="all"))
