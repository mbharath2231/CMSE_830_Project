import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Missing Values Heatmap")

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# Sidebar selector
dataset_name = st.sidebar.selectbox(
    "Choose a Dataset",
    ("Matches", "Deliveries")
)

# Select the dataset
df = matches if dataset_name == "Matches" else deliveries

st.write(f"### 🔍 Missing Values Heatmap — {dataset_name} Dataset")

# Create Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
ax.set_title(f"Missing Values Heatmap - {dataset_name}", fontsize=15)

# Display in Streamlit
st.pyplot(fig)
