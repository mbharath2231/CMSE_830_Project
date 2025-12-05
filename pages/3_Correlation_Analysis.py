import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Correlation Analysis", layout="wide")

st.header("📊 Correlation Heatmap")
st.markdown("""
This chart shows how different numeric variables relate to each other.
* **1.0**: Perfect positive correlation (Both go up together).
* **-1.0**: Perfect negative correlation (One goes up, the other goes down).
""")

# 1. LOAD DATA
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    return matches, deliveries

try:
    matches, deliveries = load_data()
except FileNotFoundError:
    st.error("❌ Data files not found. Please check your 'data' folder.")
    st.stop()

# 2. SELECT DATASET
dataset_option = st.selectbox(
    "Select Dataset for Correlation Heatmap",
    ("Matches Dataset", "Deliveries Dataset")
)

if dataset_option == "Matches Dataset":
    df = matches
else:
    df = deliveries

st.subheader(f"Dataset Selected: {dataset_option}")

# 3. PREPROCESSING (The Fix for Empty Cells)
# Step A: Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Step B: Drop columns that are IDs (usually useless for correlation)
# We look for columns ending in 'id' or specific known ID columns
cols_to_drop = [c for c in numeric_df.columns if 'id' in c.lower()]
numeric_df = numeric_df.drop(columns=cols_to_drop, errors='ignore')

# Step C: Fill Missing Values
# Correlation fails if there are NaNs. We fill with 0 or Median.
numeric_df = numeric_df.fillna(0)

# Step D: Remove "Constant" Columns (The Root Cause of Empty Cells)
# If a column has a standard deviation of 0 (all values are the same), correlation is Impossible (NaN).
# We filter these out.
numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

# 4. VISUALIZATION
if numeric_df.shape[1] < 2:
    st.warning("⚠️ Not enough numeric data with variance to create a heatmap.")
else:
    # Calculate Correlation
    corr = numeric_df.corr()

    # Plotly Heatmap
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red-Blue is better for +/- correlation
        zmin=-1, zmax=1,                 # Lock scale between -1 and 1
        labels=dict(color="Correlation"),
        title=f"Correlation Heatmap - {dataset_option}",
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.info("ℹ️ **Note:** Columns with only one unique value (Zero Variance) or ID columns have been removed to prevent empty cells.")