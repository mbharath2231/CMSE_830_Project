import streamlit as st
import pandas as pd
import plotly.express as px

st.header("📊 Correlation Heatmap")

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# Page selector
dataset_option = st.selectbox(
    "Select Dataset for Correlation Heatmap",
    ("Matches Dataset", "Deliveries Dataset")
)

# Choose dataset
if dataset_option == "Matches Dataset":
    df = matches
else:
    df = deliveries

st.subheader(f"Dataset Selected: {dataset_option}")

# Select numeric columns only
numeric_df = df.select_dtypes(include=['int64', 'float64'])

if numeric_df.shape[1] < 2:
    st.warning("Not enough numeric columns to create a correlation heatmap.")
else:
    corr = numeric_df.corr()

    # Plotly heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(color="Correlation"),
        title=f"Correlation Heatmap - {dataset_option}",
    )

    st.plotly_chart(fig, use_container_width=True)
