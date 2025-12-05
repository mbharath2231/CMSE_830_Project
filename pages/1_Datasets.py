import streamlit as st
import pandas as pd
import io

# 1. PAGE CONFIG
st.set_page_config(page_title="Dataset Overview", layout="wide")

st.title("🗂️ Data Inventory")
st.markdown("""
**Overview of the raw data sources used in this project.** Use this page to audit the data quality, check dimensions, and download samples.
""")

# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("data/matches.csv")
        deliveries = pd.read_csv("data/deliveries.csv")
        return matches, deliveries
    except FileNotFoundError:
        st.error("❌ Data files not found. Please ensure they are in the 'data/' folder.")
        return None, None

matches, deliveries = load_data()

if matches is not None:
    
    # 3. HELPER FUNCTION FOR RICH INFO
    def dataset_snapshot(df, name):
        """Displays high-level stats for a dataframe"""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            # Memory usage in MB
            mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{mem:.2f} MB")
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicates", duplicates, delta_color="inverse" if duplicates > 0 else "off")

    # 4. TABS
    tab1, tab2 = st.tabs(["📘 Matches Data", "📗 Deliveries Data"])

    # ==========================
    # TAB 1: MATCHES
    # ==========================
    with tab1:
        st.subheader("Matches Overview")
        dataset_snapshot(matches, "Matches")
        
        st.divider()
        
        # A. INTERACTIVE PREVIEW
        col_opt, col_view = st.columns([1, 4])
        with col_opt:
            view_mode = st.radio("View Mode", ["Head (5)", "Tail (5)", "Random Sample (10)"], key="m_view")
            
        with col_view:
            if "Head" in view_mode:
                display_df = matches.head(5)
            elif "Tail" in view_mode:
                display_df = matches.tail(5)
            else:
                display_df = matches.sample(10)
            
            st.dataframe(display_df, use_container_width=True)

        # B. DATA QUALITY AUDIT (The "Cool" Part)
        st.subheader("🛡️ Data Quality Audit")
        
        # Calculate missing values
        missing = matches.isnull().sum()
        missing = missing[missing > 0]
        
        c1, c2 = st.columns(2)
        with c1:
            if not missing.empty:
                st.write("**Missing Values Heatmap**")
                # Create a simple dataframe for the bar chart
                miss_df = pd.DataFrame({'Column': missing.index, 'Count': missing.values})
                st.dataframe(
                    miss_df,
                    column_config={
                        "Count": st.column_config.ProgressColumn(
                            "Missing Count",
                            format="%d",
                            min_value=0,
                            max_value=matches.shape[0],
                            help="Number of missing values",
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ No missing values detected in this dataset!")

        with c2:
            st.write("**Column Data Types**")
            # Group columns by type
            types = matches.dtypes.value_counts()
            st.bar_chart(types)

    # ==========================
    # TAB 2: DELIVERIES
    # ==========================
    with tab2:
        st.subheader("Deliveries Overview")
        dataset_snapshot(deliveries, "Deliveries")
        
        st.divider()
        
        # A. INTERACTIVE PREVIEW
        col_opt, col_view = st.columns([1, 4])
        with col_opt:
            view_mode_d = st.radio("View Mode", ["Head (5)", "Tail (5)", "Random Sample (10)"], key="d_view")
            
        with col_view:
            if "Head" in view_mode_d:
                disp_d = deliveries.head(5)
            elif "Tail" in view_mode_d:
                disp_d = deliveries.tail(5)
            else:
                disp_d = deliveries.sample(10)
            
            st.dataframe(disp_d, use_container_width=True)

        # B. DATA QUALITY AUDIT
        st.subheader("🛡️ Data Quality Audit")
        
        missing_d = deliveries.isnull().sum()
        missing_d = missing_d[missing_d > 0]
        
        c3, c4 = st.columns(2)
        with c3:
            if not missing_d.empty:
                st.write("**Missing Values Heatmap**")
                miss_df_d = pd.DataFrame({'Column': missing_d.index, 'Count': missing_d.values})
                st.dataframe(
                    miss_df_d,
                    column_config={
                        "Count": st.column_config.ProgressColumn(
                            "Missing Count",
                            format="%d",
                            min_value=0,
                            max_value=deliveries.shape[0],
                            help="Number of missing values",
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ No missing values detected!")

        with c4:
            st.write("**Key Statistics (Numeric)**")
            # Show describe only for important numeric cols to avoid clutter
            important_cols = ['runs_off_bat', 'extras', 'wide_runs', 'noball_runs']
            # Filter cols that actually exist in the dataframe
            existing_cols = [c for c in important_cols if c in deliveries.columns]
            if existing_cols:
                st.dataframe(deliveries[existing_cols].describe(), use_container_width=True)