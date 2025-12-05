import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Player Clustering", layout="wide")

st.title("🧩 Squad Role Analysis")
st.markdown("""
**Goal:** Identify player roles (e.g., "Anchors" vs "Power Hitters") using Unsupervised Machine Learning.
Select a mode below to analyze the squad.
""")

# 2. SHARED DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/deliveries.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure 'data/deliveries.csv' exists.")
        return None

df = load_data()

if df is not None:
    # 3. ANALYSIS TOGGLE
    st.divider()
    analysis_type = st.radio(
        "Select Analysis Mode:",
        ["🏏 Batsmen Clustering", "🎯 Bowler Clustering"],
        horizontal=True
    )

    # ==========================================
    # LOGIC 1: BATSMEN CLUSTERING
    # ==========================================
    if analysis_type == "🏏 Batsmen Clustering":
        st.subheader("1. Configure Parameters")
        
        # A. Preprocessing
        player_stats = df.groupby('striker').agg(
            runs=('runs_off_bat', 'sum'),
            balls=('ball', 'count'),
            dismissals=('player_dismissed', lambda x: x.notnull().sum())
        ).reset_index()

        # B. Controls with Embedded Docs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### **Step A: Filter Data**")
            min_balls = st.slider(
                "Minimum Balls Faced", 
                10, 200, 50, 10, 
                key="bat_balls",
                help="We remove players with low ball counts to avoid statistical noise (e.g., a player who hit 1 six and never played again)."
            )
            st.caption("ℹ️ **Why?** Removes outliers. A player needs a decent sample size to determine their true 'style'.")

        with col2:
            st.markdown("##### **Step B: ML Settings**")
            k = st.slider(
                "Number of Clusters (k)", 
                2, 6, 4, 
                key="bat_k",
                help="K-Means algorithm will try to find this many distinct groups in the data."
            )
            st.caption("ℹ️ **What is k?** The number of 'roles' you want the AI to find (e.g., 3 = Low, Mid, High performance).")

        # C. Filter
        filtered_players = player_stats[player_stats['balls'] >= min_balls].copy()
        
        if len(filtered_players) < k:
             st.error("Not enough players match your filter. Lower the 'Minimum Balls'.")
        else:
            # D. Metrics
            filtered_players['strike_rate'] = (filtered_players['runs'] / filtered_players['balls']) * 100
            filtered_players['average'] = filtered_players.apply(
                lambda x: x['runs'] / x['dismissals'] if x['dismissals'] > 0 else x['runs'], axis=1
            )

            # E. AI Model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            features = filtered_players[['average', 'strike_rate']]
            filtered_players['cluster'] = kmeans.fit_predict(features)
            filtered_players['cluster_label'] = filtered_players['cluster'].astype(str)

            # F. Visualization
            st.divider()
            st.subheader("2. Cluster Visualization")
            
            # Context for the Chart
            st.info("""
            **How to read this chart:**
            * **X-Axis (Batting Avg):** Consistency. (Runs per out).
            * **Y-Axis (Strike Rate):** Aggression. (Runs per 100 balls).
            * **Goal:** Look for players in the **Top-Right** (High Avg + High SR).
            """)

            fig = px.scatter(
                filtered_players, x='average', y='strike_rate', color='cluster_label',
                hover_name='striker', size='runs', 
                title=f"identified {k} Distinct Batting Styles",
                labels={'average': 'Batting Average', 'strike_rate': 'Strike Rate'},
                height=600, template="plotly_white"
            )
            fig.add_hline(y=filtered_players['strike_rate'].mean(), line_dash="dash", line_color="grey", annotation_text="Avg SR")
            fig.add_vline(x=filtered_players['average'].mean(), line_dash="dash", line_color="grey", annotation_text="Avg Avg")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # G. Stats Table
            st.write("### 3. Group Summaries")
            summary = filtered_players.groupby('cluster_label').agg(
                Avg_Strike_Rate=('strike_rate', 'mean'),
                Avg_Average=('average', 'mean'),
                Player_Count=('striker', 'count'),
                Examples=('striker', lambda x: ", ".join(x.sample(min(3, len(x))).values))
            ).sort_values('Avg_Average', ascending=False)
            st.dataframe(summary.style.format("{:.1f}", subset=['Avg_Strike_Rate', 'Avg_Average']), use_container_width=True)

    # ==========================================
    # LOGIC 2: BOWLER CLUSTERING
    # ==========================================
    elif analysis_type == "🎯 Bowler Clustering":
        st.subheader("1. Configure Parameters")

        # A. Preprocessing
        if 'runs_off_bat' in df.columns:
            df['bowler_runs_conceded'] = df['runs_off_bat']
            if 'wide_runs' in df.columns: df['bowler_runs_conceded'] += df['wide_runs'].fillna(0)
            if 'noball_runs' in df.columns: df['bowler_runs_conceded'] += df['noball_runs'].fillna(0)
            if 'wide_runs' not in df.columns and 'extras' in df.columns: df['bowler_runs_conceded'] += df['extras'].fillna(0)
        else:
             st.error("Critical: 'runs_off_bat' column missing.")
             st.stop()
        
        df['is_legal_ball'] = 1
        if 'wide_runs' in df.columns: df.loc[df['wide_runs'] > 0, 'is_legal_ball'] = 0
        if 'noball_runs' in df.columns: df.loc[df['noball_runs'] > 0, 'is_legal_ball'] = 0
        
        valid_dismissals = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
        if 'dismissal_kind' in df.columns:
            df['is_bowler_wicket'] = df['dismissal_kind'].isin(valid_dismissals).astype(int)
        else:
            df['is_bowler_wicket'] = df['player_dismissed'].notnull().astype(int)

        bowler_stats = df.groupby('bowler').agg(
            total_runs=('bowler_runs_conceded', 'sum'),
            legal_balls=('is_legal_ball', 'sum'),
            total_wickets=('is_bowler_wicket', 'sum')
        ).reset_index()
        bowler_stats['overs_bowled'] = bowler_stats['legal_balls'] / 6

        # B. Controls with Embedded Docs
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **Step A: Filter Data**")
            min_overs = st.slider(
                "Minimum Overs Bowled", 
                1, 50, 5, 1, 
                key="bowl_overs",
                help="Removes 'Part-Timers' who only bowled 1 or 2 overs, as their stats are often misleading."
            )
            st.caption("ℹ️ **Why?** We need consistent bowlers. 5 overs is roughly 2 games worth of data.")

        with col2:
            st.markdown("##### **Step B: ML Settings**")
            k = st.slider(
                "Number of Clusters (k)", 
                2, 5, 3, 
                key="bowl_k",
                help="The number of distinct bowling strategies you want to identify."
            )
            st.caption("ℹ️ **Recommendation:** k=3 usually separates 'Economical', 'Wicket Takers', and 'Expensive'.")

        # C. Filter
        filtered_bowlers = bowler_stats[bowler_stats['overs_bowled'] >= min_overs].copy()
        
        filtered_bowlers['economy'] = filtered_bowlers['total_runs'] / filtered_bowlers['overs_bowled']
        filtered_bowlers = filtered_bowlers[filtered_bowlers['total_wickets'] > 0]
        
        if len(filtered_bowlers) < k:
             st.error(f"Not enough bowlers ({len(filtered_bowlers)}) found. Try lowering 'Minimum Overs'.")
        else:
            filtered_bowlers['average'] = filtered_bowlers['total_runs'] / filtered_bowlers['total_wickets']

            # E. AI Model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            features = filtered_bowlers[['economy', 'average']]
            filtered_bowlers['cluster'] = kmeans.fit_predict(features)
            filtered_bowlers['cluster_label'] = filtered_bowlers['cluster'].astype(str)

            # F. Visualization
            st.divider()
            st.subheader("2. Cluster Visualization")

            # Context for the Chart
            st.info("""
            **How to read this chart:**
            * **X-Axis (Economy):** Runs conceded per Over. (Lower is better).
            * **Y-Axis (Bowling Avg):** Runs conceded per Wicket. (Lower is better).
            * **Goal:** Look for players in the **Bottom-Left** (Low Econ + Low Avg).
            """)

            fig = px.scatter(
                filtered_bowlers, x='economy', y='average', color='cluster_label',
                size='total_wickets', hover_name='bowler', 
                title=f"Identified {k} Bowling Strategies",
                labels={'economy': 'Economy Rate', 'average': 'Bowling Average'},
                height=600, template="plotly_white"
            )
            fig.add_hline(y=filtered_bowlers['average'].mean(), line_dash="dash", line_color="grey", annotation_text="Avg")
            fig.add_vline(x=filtered_bowlers['economy'].mean(), line_dash="dash", line_color="grey", annotation_text="Econ")
            
            st.plotly_chart(fig, use_container_width=True)

            # G. Stats Table
            st.write("### 3. Group Summaries")
            summary = filtered_bowlers.groupby('cluster_label').agg(
                Avg_Economy=('economy', 'mean'),
                Avg_Average=('average', 'mean'),
                Player_Count=('bowler', 'count'),
                Examples=('bowler', lambda x: ", ".join(x.sample(min(3, len(x))).values))
            ).sort_values('Avg_Average')
            st.dataframe(summary.style.format("{:.2f}", subset=['Avg_Economy', 'Avg_Average']), use_container_width=True)