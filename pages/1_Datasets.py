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
    st.markdown("""
    This table shows the raw match-level dataset exactly as collected.
    It includes information such as match ID, teams, venue, toss decision,
    and winner. Viewing the unprocessed data helps verify data integrity
    and understand the structure before performing analysis.
    """)
    with st.expander("Show Matches Dataframe"):
        st.dataframe(matches)

    st.subheader("📐 Shape")
    st.markdown("""
    The shape tells how many rows (matches) and columns (attributes)
    the dataset contains. This helps estimate dataset size and complexity,
    and confirms whether all records were loaded successfully.
    """)
    st.write(f"Rows: **{matches.shape[0]}**, Columns: **{matches.shape[1]}**")

    st.subheader("🔤 Column Data Types")
    st.markdown("""
    Each column has a specific data type (numeric, categorical, object).
    Understanding data types is important for EDA and modeling because
    certain analyses (like correlation) only work on numeric features,
    while others require categorical encoding.
    """)
    st.dataframe(matches.dtypes.rename("dtype"))

    st.subheader("📊 Missing Value Summary")
    st.markdown("""
    This section displays how many values are missing in each column.
    Missing data may arise from incomplete scorecards or unavailable
    information. Identifying missing values early helps determine whether
    imputation or removal is required before analysis or modeling.
    """)
    st.dataframe(matches.isnull().sum().rename("missing_values"))

    st.subheader("📈 Descriptive Statistics")
    st.markdown("""
    This provides summary statistics such as count, mean, standard
    deviation, minimum, maximum, and unique values. It helps identify
    variable ranges, detect anomalies, and understand distributions.
    Descriptive stats act as the foundation for deeper exploratory analysis.
    """)
    st.dataframe(matches.describe(include="all"))

# -----------------------------
# TAB 2 — DELIVERIES
# -----------------------------
with tab2:
    st.subheader("🔍 Raw Data Preview")
    st.markdown("""
    The deliveries dataset contains ball-by-ball information for every match.
    It is the most detailed level of cricket data, capturing runs scored,
    extras, wickets, bowler, striker, and over/ball numbers. Examining this
    preview helps ensure that granular data required for advanced analytics
    is available.
    """)
    with st.expander("Show Deliveries Dataframe"):
        st.dataframe(deliveries)

    st.subheader("📐 Shape")
    st.markdown("""
    This shows the number of ball entries and the number of recorded features
    per delivery. Since T20 matches produce thousands of rows per tournament,
    this dataset is naturally much larger than the match-level file.
    """)
    st.write(f"Rows: **{deliveries.shape[0]}**, Columns: **{deliveries.shape[1]}**")

    st.subheader("🔤 Column Data Types")
    st.markdown("""
    Column data types help identify which fields can be used for numeric
    aggregation (e.g., runs_off_bat), which are categorical (e.g., bowler,
    striker), and which require preprocessing. This is important for feature
    engineering in modeling tasks.
    """)
    st.dataframe(deliveries.dtypes.rename("dtype"))

    st.subheader("📊 Missing Value Summary")
    st.markdown("""
    Any missing values in the deliveries dataset may correspond to missing
    dismissal information or incomplete scorecard entries. Identifying and
    addressing missingness ensures accurate computation of metrics such
    as strike rate, economy, and powerplay statistics.
    """)
    st.dataframe(deliveries.isnull().sum().rename("missing_values"))

    st.subheader("📈 Descriptive Statistics")
    st.markdown("""
    Basic statistical summaries help understand the range and distribution
    of ball-by-ball variables such as runs, extras, and wickets. These stats
    also serve as sanity checks to ensure the scoring data aligns with expected
    T20 cricket patterns.
    """)
    st.dataframe(deliveries.describe(include="all"))
