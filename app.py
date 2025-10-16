import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title = "T20 WorldCup Dashboard", layout = "wide")

st.title("ICC T20 WorldCup Analysis - Team Insights")
st.write("Explore the Match Results, Team Performance and Match Venues")

matches = pd.read_csv("matches.csv")

teams = matches['winner'].dropna().unique()
selected_team = st.sidebar.selectbox("Select a team", options = ["All"] + list(teams))

if selected_team != "All" : 
    df = matches[matches["winner"] == selected_team]
else : 
    df = matches

# -----------------------------
# Visualization 1 - Team Wins
# -----------------------------

wins = df['winner'].value_counts().reset_index()
wins.columns = ['Team', 'Wins']

fig1 = px.bar(
    wins,
    x = 'Team',
    y = 'Wins',
    color = 'Wins',
    title = 'Number of Wins by Each Team',
    hover_name = 'Team'
)

st.plotly_chart(fig1, use_container_width = True)

fig2 = px.pie(
    df,
    names = 'toss_decision',
    title = 'Toss Decision Distribution (Bat/Field)',
    color_discrete_sequence = px.colors.sequential.RdBu
)

st.plotly_chart(fig2, use_container_width=True)