import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("🧩 Missing Data & Imputation")

@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

dataset_name = st.selectbox(
    "Choose a Dataset",
    ("Matches", "Deliveries")
)

df = matches if dataset_name == "Matches" else deliveries

st.markdown(f"### 🔍 Missing Values Heatmap — {dataset_name} Dataset")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
ax.set_title(f"Missing Values Heatmap - {dataset_name}", fontsize=15)
st.pyplot(fig)

st.markdown("---")
st.subheader("🛠 Imputation Techniques")

method = st.selectbox(
    "Choose Imputation Technique:",
    [
        "Mean Imputation",
        "Median Imputation",
        "Mode Imputation",
        "Forward Fill (ffill)",
        "Backward Fill (bfill)",
        "Drop Rows with Missing Values"
    ]
)

if st.button("Apply Imputation"):
    df_imputed = df.copy()

    if method == "Mean Imputation":
        numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].mean())

    elif method == "Median Imputation":
        numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].median())

    elif method == "Mode Imputation":
        for col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])

    elif method == "Forward Fill (ffill)":
        df_imputed = df_imputed.ffill()

    elif method == "Backward Fill (bfill)":
        df_imputed = df_imputed.bfill()

    elif method == "Drop Rows with Missing Values":
        df_imputed = df_imputed.dropna()

    # ---- Before vs After comparison ----
    st.subheader("📊 Missing Values Before vs After Imputation")

    before_missing = df.isna().sum()
    after_missing = df_imputed.isna().sum()

    comparison_df = pd.DataFrame({
        "Column": df.columns,
        "Before Imputation": before_missing.values,
        "After Imputation": after_missing.values
    })

    fig2 = go.Figure(data=[
        go.Bar(name='Before', x=comparison_df["Column"], y=comparison_df["Before Imputation"]),
        go.Bar(name='After', x=comparison_df["Column"], y=comparison_df["After Imputation"])
    ])

    fig2.update_layout(
        barmode='group',
        title="Missing Values Comparison",
        xaxis_title="Columns",
        yaxis_title="Number of Missing Values",
        height=500
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔎 Preview of Imputed Data")
    st.dataframe(df_imputed.head())
