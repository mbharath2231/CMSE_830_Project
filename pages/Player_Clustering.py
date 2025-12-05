import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# Title
st.title("🏏 Batsmen Clustering Analysis")
st.markdown("Use Machine Learning to group players into roles (e.g., Power Hitters, Anchors) based on their performance.")

# 1. Load Data
try:
    deliveries = pd.read_csv('data/deliveries.csv')
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'data/deliveries.csv' exists.")
    st.stop()

# 2. Data Preprocessing (Feature Engineering)
# Group by batsman to get total runs, balls faced, and dismissals
player_stats = deliveries.groupby('striker').agg(
    runs=('runs_off_bat', 'sum'),
    balls=('ball', 'count'),
    dismissals=('player_dismissed', lambda x: x.notnull().sum())
).reset_index()

# Sidebar Filter: Remove outliers (players who played very little)
st.sidebar.header("Clustering Parameters")
min_balls = st.sidebar.slider("Minimum Balls Faced", min_value=10, max_value=200, value=50, step=10)
filtered_players = player_stats[player_stats['balls'] >= min_balls].copy()

# Calculate Key Metrics
# Strike Rate: (Runs / Balls) * 100
filtered_players['strike_rate'] = (filtered_players['runs'] / filtered_players['balls']) * 100

# Batting Average: Runs / Dismissals (Handle division by zero)
filtered_players['average'] = filtered_players.apply(
    lambda x: x['runs'] / x['dismissals'] if x['dismissals'] > 0 else x['runs'], axis=1
)

# 3. K-Means Clustering
# User selects 'k' (number of clusters)
k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=6, value=4)

# Select features for clustering
features = filtered_players[['average', 'strike_rate']]

# Train the model
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
filtered_players['cluster'] = kmeans.fit_predict(features)

# Convert cluster to string so it's treated as a category in the plot
filtered_players['cluster_label'] = filtered_players['cluster'].astype(str)

# 4. Visualization
st.subheader(f"Player Clusters (k={k})")

fig = px.scatter(
    filtered_players,
    x='average',
    y='strike_rate',
    color='cluster_label',
    hover_name='striker',
    size='runs',  # Bubble size represents total runs
    labels={
        'average': 'Batting Average',
        'strike_rate': 'Strike Rate (%)',
        'cluster_label': 'Cluster Group'
    },
    title="Batting Average vs. Strike Rate (Colored by Cluster)",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 5. Cluster Interpretation (Explain what the groups mean)
st.subheader("Cluster Insights")
st.write("Here are the average stats for each group. You can likely identify them as 'Power Hitters', 'Anchors', etc.")

summary = filtered_players.groupby('cluster_label').agg(
    Avg_Strike_Rate=('strike_rate', 'mean'),
    Avg_Batting_Avg=('average', 'mean'),
    Player_Count=('striker', 'count'),
    Example_Players=('striker', lambda x: ", ".join(x.sample(min(3, len(x))).values)) # Show 3 random players
).reset_index()

st.dataframe(summary.style.format({"Avg_Strike_Rate": "{:.2f}", "Avg_Batting_Avg": "{:.2f}"}))