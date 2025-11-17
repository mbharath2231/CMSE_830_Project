import streamlit as st
import pandas as pd

st.title("📊 Dataset Summary")

st.write("""
This page provides a comprehensive summary of the **Matches** and **Deliveries**
datasets used in the project.  
It includes dataset size, column data types, descriptive statistics,  
and missing value summaries.
""")

# Load datasets (Streamlit will use cached version if placed in main app)
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# --- TAB LAYOUT FOR CLEANER UI ---
tab1, tab2 = st.tabs(["📘 Matches Dataset", "📗 Deliveries Dataset"])

# -----------------------------
# TAB 1 — MATCHES DATASET
# -----------------------------
with tab1:
    st.header("📘 Matches Dataset")

    st.subheader("📐 Shape")
    st.write(f"Rows: **{matches.shape[0]}**, Columns: **{matches.shape[1]}**")

    st.subheader("🔤 Column Data Types")
    st.dataframe(matches.dtypes.rename("dtype"))

    st.subheader("📊 Missing Value Summary")
    st.dataframe(matches.isnull().sum().rename("missing_values"))

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(matches.describe(include="all"))

# -----------------------------
# TAB 2 — DELIVERIES DATASET
# -----------------------------
with tab2:
    st.header("📗 Deliveries Dataset")

    st.subheader("📐 Shape")
    st.write(f"Rows: **{deliveries.shape[0]}**, Columns: **{deliveries.shape[1]}**")

    st.subheader("🔤 Column Data Types")
    st.dataframe(deliveries.dtypes.rename("dtype"))

    st.subheader("📊 Missing Value Summary")
    st.dataframe(deliveries.isnull().sum().rename("missing_values"))

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(deliveries.describe(include="all"))
