import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Player Clustering", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧩 Squad Role Analysis & PCA")
st.markdown("""
**Goal:** Identify player roles (e.g., "Anchors" vs "Power Hitters") using **Unsupervised Machine Learning (K-Means)**.
We also use **PCA (Principal Component Analysis)** to visualize these multi-dimensional styles in 2D space.
""")

# 2. DATA LOADING & PREP
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/deliveries.csv')
        
        # Standardize column names
        if 'wides' in df.columns:
            df['wide_runs'] = df['wides'].fillna(0)
        elif 'wide_runs' in df.columns:
            df['wide_runs'] = df['wide_runs'].fillna(0)
        else:
            df['wide_runs'] = 0
            
        if 'noballs' in df.columns:
            df['noball_runs'] = df['noballs'].fillna(0)
        elif 'noball_runs' in df.columns:
            df['noball_runs'] = df['noball_runs'].fillna(0)
        else:
            df['noball_runs'] = 0
            
        # Total runs calculation
        if 'total_runs' not in df.columns:
            if 'runs_off_bat' in df.columns and 'extras' in df.columns:
                df['total_runs'] = df['runs_off_bat'] + df['extras']
            else:
                st.error("❌ Critical Error: Missing 'runs_off_bat' or 'extras'.")
                return None
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    st.divider()
    mode = st.radio("Select Analysis Mode:", ["🏏 Batsmen Clustering", "🎯 Bowler Clustering"], horizontal=True)

    # ==========================================
    # LOGIC 1: BATSMEN CLUSTERING
    # ==========================================
    if mode == "🏏 Batsmen Clustering":
        # Feature Engineering
        stats = df.groupby('striker').agg(
            runs=('runs_off_bat', 'sum'),
            balls=('ball', 'count'),
            fours=('runs_off_bat', lambda x: (x==4).sum()),
            sixes=('runs_off_bat', lambda x: (x==6).sum()),
            dismissals=('player_dismissed', lambda x: x.notnull().sum())
        ).reset_index()
        
        # User Inputs (Updated Ranges)
        col1, col2 = st.columns(2)
        # UPDATED: Increased max range to 300, set default to 30 to catch high-SR hitters
        min_balls = col1.slider("Filter: Min Balls Faced", 10, 300, 30, help="Lower this to see 'Finishers' who face fewer balls but score fast.")
        k = col2.slider("K-Means Clusters", 2, 6, 4)
        
        # Filtering
        filtered = stats[stats['balls'] >= min_balls].copy()
        
        # Metrics
        filtered['strike_rate'] = filtered['runs'] / filtered['balls'] * 100
        filtered['average'] = filtered.apply(lambda x: x['runs']/x['dismissals'] if x['dismissals']>0 else x['runs'], axis=1)
        filtered['boundary_pct'] = (filtered['fours']*4 + filtered['sixes']*6) / filtered['runs'] * 100
        filtered = filtered.fillna(0)
        
        features = ['runs', 'strike_rate', 'average', 'boundary_pct']
        
        # Clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(filtered[features])
        kmeans = KMeans(n_clusters=k, random_state=42)
        filtered['cluster'] = kmeans.fit_predict(X_scaled).astype(str)
        
        # --- VISUALIZATION ---
        st.subheader("1. Identify Batting Styles")
        fig = px.scatter(
            filtered, x='average', y='strike_rate', color='cluster', 
            size='runs', hover_name='striker',
            title=f"Batting Roles (k={k})",
            labels={'average': 'Batting Average', 'strike_rate': 'Strike Rate'},
            template="plotly_white"
        )
        # Ensure we see high Strike Rates
        fig.update_yaxes(range=[0, max(filtered['strike_rate'].max() * 1.1, 200)])
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # LOGIC 2: BOWLER CLUSTERING (FIXED)
    # ==========================================
    else:
        # Determine Legal Balls
        df['is_legal'] = 1
        df.loc[df['wide_runs'] > 0, 'is_legal'] = 0
        df.loc[df['noball_runs'] > 0, 'is_legal'] = 0
        
        # Determine Bowler Wickets
        valid_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
        if 'wicket_type' in df.columns:
            df['is_wicket'] = df['wicket_type'].isin(valid_wickets).astype(int)
        else:
            df['is_wicket'] = df['player_dismissed'].notnull().astype(int)

        # Feature Engineering
        stats = df.groupby('bowler').agg(
            runs_conceded=('total_runs', 'sum'),
            legal_balls=('is_legal', 'sum'),
            wickets=('is_wicket', 'sum')
        ).reset_index()
        
        stats['overs'] = stats['legal_balls'] / 6
        
        # User Inputs
        col1, col2 = st.columns(2)
        min_overs = col1.slider("Filter: Min Overs Bowled", 1, 60, 10)
        k = col2.slider("K-Means Clusters", 2, 6, 3)
        
        # Filtering
        filtered = stats[stats['overs'] >= min_overs].copy()
        
        if len(filtered) < k:
            st.error("Not enough bowlers match your filter. Lower the 'Min Overs'.")
            st.stop()
            
        # Metrics
        filtered['economy'] = filtered['runs_conceded'] / filtered['overs']
        # UPDATED: Increased penalty for 0-wicket bowlers to 150 (Visual separation)
        filtered['strike_rate'] = filtered.apply(lambda x: x['legal_balls']/x['wickets'] if x['wickets']>0 else 150, axis=1) 
        filtered['average'] = filtered.apply(lambda x: x['runs_conceded']/x['wickets'] if x['wickets']>0 else 60, axis=1)
        
        features = ['economy', 'strike_rate', 'average']
        
        # Clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(filtered[features])
        kmeans = KMeans(n_clusters=k, random_state=42)
        filtered['cluster'] = kmeans.fit_predict(X_scaled).astype(str)
        
        # --- VISUALIZATION ---
        st.subheader("1. Identify Bowling Strategies")
        fig = px.scatter(
            filtered, x='economy', y='strike_rate', color='cluster',
            size='wickets', hover_name='bowler',
            title=f"Bowling Roles (k={k})",
            labels={'economy': 'Economy Rate (Runs/Over)', 'strike_rate': 'Strike Rate (Balls/Wicket)'},
            template="plotly_white"
        )
        # Note: In Bowling, Lower is Better for both axes
        fig.update_yaxes(autorange="reversed") 
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: Axes are reversed because lower Economy and lower Strike Rate are better.")

    # ==========================================
    # PCA (DIMENSIONALITY REDUCTION)
    # ==========================================
    st.divider()
    st.subheader("Dimensionality Reduction (PCA)")
    
    col_pca1, col_pca2 = st.columns([1, 2])
    
    with col_pca1:
        st.info("""
        **Why PCA?**
        Our players have multiple stats (Avg, SR, Boundaries, etc.). It's hard to visualize 4D data.
        
        **Principal Component Analysis (PCA)** compresses these into 2 main "Components" while keeping the variance.
        * **PC1:** Usually represents overall "Quality" or "Performance".
        * **PC2:** Usually represents "Style" (Aggressive vs Defensive).
        """)
        
    with col_pca2:
        # Run PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_scaled)
        
        filtered['PC1'] = pca_components[:, 0]
        filtered['PC2'] = pca_components[:, 1]
        
        # Plot PCA
        fig_pca = px.scatter(
            filtered, x='PC1', y='PC2', color='cluster',
            hover_name=filtered.columns[0], # Striker or Bowler
            title=f"PCA Projection (Variance Explained: {sum(pca.explained_variance_ratio_):.1%})",
            template="plotly_white"
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
    st.success("✅ **Interpretation:** Players grouped together in this PCA plot are statistically similar in *all* dimensions, not just the ones shown in the first chart.")