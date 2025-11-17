import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 

# ---------------------------------
# 4. MISSING VALUE IMPUTATION
# ---------------------------------
st.subheader("🛠 Missing Value Imputation")

st.write("Apply imputation methods to handle missing values:")

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

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

dataset_name = st.sidebar.selectbox(
    "Choose a Dataset",
    ("Matches", "Deliveries")
)

# Select the dataset
df = matches if dataset_name == "Matches" else deliveries

df_imputed = df.copy()

if st.button("Apply Imputation"):

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

    # --------------------------------------------------
    #  MISSING VALUE COMPARISON PLOT (BEFORE VS AFTER)
    # --------------------------------------------------
    st.subheader("📊 Missing Values Before vs After Imputation")

    before_missing = df.isna().sum()
    after_missing = df_imputed.isna().sum()

    comparison_df = pd.DataFrame({
        "Column": df.columns,
        "Before Imputation": before_missing.values,
        "After Imputation": after_missing.values
    })

    # Interactive bar chart
    fig = go.Figure(data=[
        go.Bar(name='Before', x=comparison_df["Column"], y=comparison_df["Before Imputation"]),
        go.Bar(name='After', x=comparison_df["Column"], y=comparison_df["After Imputation"])
    ])

    fig.update_layout(
        barmode='group',
        title="Missing Values Comparison",
        xaxis_title="Columns",
        yaxis_title="Number of Missing Values",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_imputed.head())
