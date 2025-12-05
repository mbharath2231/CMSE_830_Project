import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Data Cleaning", layout="wide")

st.title("🧹 Data Preprocessing & Imputation")
st.markdown("""
**Goal:** Prepare your data for Machine Learning by fixing missing values (`NaN`).
Follow the 3-step process below: **Audit** -> **Configure** -> **Repair**.
""")

# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("data/matches.csv")
        deliveries = pd.read_csv("data/deliveries.csv")
        return matches, deliveries
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure 'data/matches.csv' and 'data/deliveries.csv' exist.")
        return None, None

matches, deliveries = load_data()

if matches is not None:
    # ---------------------------------------------------------
    # STEP 1: CONFIGURATION (MAIN PAGE - NO SIDEBAR)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("1. Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### **Select Dataset**")
        dataset_name = st.selectbox(
            "Target File",
            ("Matches", "Deliveries"),
            label_visibility="collapsed",
            help="Choose which CSV file you want to analyze and clean."
        )
        st.caption("ℹ️ **Context:** 'Matches' contains game info (teams, venue). 'Deliveries' contains ball-by-ball data.")

    # Select the dataframe
    df = matches if dataset_name == "Matches" else deliveries
    
    with col2:
        st.markdown("##### **Select Imputation Method**")
        method = st.selectbox(
            "Method",
            [
                "Mode Imputation (Best for All)",
                "Mean Imputation (Numeric Only)",
                "Median Imputation (Numeric Only)", 
                "Forward Fill (Time Series)",
                "Drop Rows (Cleanest)"
            ],
            label_visibility="collapsed",
            help="The algorithm used to fill the empty cells."
        )
        
        # DYNAMIC DOCUMENTATION (Updates based on selection)
        if "Mean" in method:
            st.caption("ℹ️ **Logic:** Fills with Average. **Pros:** Simple. **Cons:** Ignores text columns.")
        elif "Median" in method:
            st.caption("ℹ️ **Logic:** Fills with Middle value. **Pros:** Good for skewed data. **Cons:** Ignores text.")
        elif "Mode" in method:
            st.caption("ℹ️ **Logic:** Fills with Most Frequent. **Pros:** Works on EVERYTHING (Text + Numbers).")
        elif "Forward" in method:
            st.caption("ℹ️ **Logic:** Carries previous value forward. **Pros:** Good for trends.")
        elif "Drop" in method:
            st.caption("ℹ️ **Logic:** Deletes the row entirely. **Pros:** 100% Real data. **Cons:** Data loss.")

    # ---------------------------------------------------------
    # STEP 2: AUDIT (VISUALIZE MISSING DATA BEFORE ACTION)
    # ---------------------------------------------------------
    st.divider()
    st.subheader(f"2. Audit: Where is the missing data in '{dataset_name}'?")
    
    # Calculate missing stats
    missing_series = df.isna().sum()
    missing_df = pd.DataFrame({'Column': missing_series.index, 'Missing Values': missing_series.values})
    # Filter to show only columns with > 0 missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=True)

    c1, c2 = st.columns([2, 1])
    
    with c1:
        if missing_df.empty:
            st.success("🎉 Incredible! This dataset has **ZERO** missing values. No cleaning needed.")
        else:
            # Heatmap-style Bar Chart
            fig = px.bar(
                missing_df, 
                y='Column', 
                x='Missing Values', 
                orientation='h',
                title="Missing Values per Column (Raw Data)",
                text='Missing Values',
                color='Missing Values',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.info("""
        **How to read this:**
        * The **Red Bars** show which columns are broken.
        * **Long Bars** = Lots of missing data.
        * **Tip:** If a column is 100% missing, 'Mode Imputation' will assume it is 'Unknown'.
        """)

    # ---------------------------------------------------------
    # STEP 3: EXECUTE & VERIFY
    # ---------------------------------------------------------
    st.divider()
    st.subheader("3. Apply & Verify Results")

    if st.button("🚀 Run Imputation", type="primary", use_container_width=True):
        
        df_imputed = df.copy()
        numeric_cols = df_imputed.select_dtypes(include=['number']).columns
        
        # --- APPLY LOGIC ---
        if "Mean" in method:
            if not numeric_cols.empty:
                df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].mean())
            else:
                st.warning("⚠️ No numeric columns to fill with Mean.")
                
        elif "Median" in method:
            if not numeric_cols.empty:
                df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].median())
            else:
                st.warning("⚠️ No numeric columns to fill with Median.")

        elif "Mode" in method:
            # FIX FOR KEYERROR: 0
            for col in df_imputed.columns:
                if df_imputed[col].isnull().sum() > 0:
                    # Attempt to find the mode
                    mode_values = df_imputed[col].mode()
                    
                    if not mode_values.empty:
                        # Standard Case: Fill with mode
                        df_imputed[col] = df_imputed[col].fillna(mode_values[0])
                    else:
                        # Edge Case: Column is 100% Empty (No Mode exists)
                        # We must fill it manually to prevent crash
                        if pd.api.types.is_numeric_dtype(df_imputed[col]):
                            df_imputed[col] = df_imputed[col].fillna(0)
                        else:
                            df_imputed[col] = df_imputed[col].fillna("Unknown")

        elif "Forward" in method:
            df_imputed = df_imputed.ffill()

        elif "Drop" in method:
            df_imputed = df_imputed.dropna()

        # --- VERIFICATION CHART ---
        st.markdown("##### **Outcome: Before vs After**")
        
        # Re-calculate missing after
        after_missing = df_imputed.isna().sum()
        
        # Combine for plotting
        target_cols = missing_df['Column'].tolist()
        
        if not target_cols:
             st.write("No missing values to begin with.")
        else:
            comparison_data = []
            for col in target_cols:
                comparison_data.append({'Column': col, 'Count': missing_series[col], 'State': 'Before (Red)'})
                comparison_data.append({'Column': col, 'Count': after_missing[col], 'State': 'After (Green)'})
            
            comp_df = pd.DataFrame(comparison_data)

            fig_compare = px.bar(
                comp_df,
                x='Column',
                y='Count',
                color='State',
                barmode='group',
                color_discrete_map={'Before (Red)': '#FF4B4B', 'After (Green)': '#00CC96'},
                title=f"Effect of '{method}' on Data Quality"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Final Success Message
            remaining = after_missing.sum()
            if remaining == 0:
                st.success(f"✅ CLEANING COMPLETE! All missing values in {dataset_name} have been handled.")
            else:
                st.warning(f"⚠️ Partial Cleaning: {remaining} values are still missing. (Mean/Median only fix numbers. Use 'Mode' to fix text).")