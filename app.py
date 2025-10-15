import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ICC T20 World Cup 2024 - Data Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

st.title("🏏 ICC T20 World Cup 2024 Data Analysis Dashboard")

st.sidebar.header("🔍 Filter Options")
selected_season = st.sidebar.selectbox("Select Season", sorted(matches['season'].unique()))

# --- Section 3: Top Winning Teams ---
st.subheader("🥇 Top Winning Teams")

top_teams = matches['winner'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(y=top_teams.index, x=top_teams.values, palette="crest", ax=ax)
ax.set_xlabel("Wins")
ax.set_ylabel("Team")
st.pyplot(fig)

st.subheader("🏏 Top 10 Batsmen (All Seasons)")

batsman_runs = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(y=batsman_runs.index, x=batsman_runs.values, palette="viridis", ax=ax)
ax.set_xlabel("Total Runs")
ax.set_ylabel("Batsman")
st.pyplot(fig)