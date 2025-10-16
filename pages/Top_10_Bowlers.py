import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Top Bowlers: Wickets and Economy Rate")

deliveries = pd.read_csv('data/deliveries.csv')

# Calculate wickets per bowler
wickets = deliveries[deliveries['player_dismissed'].notna()]
bowler_wickets = wickets.groupby('bowler').size().reset_index(name='wickets')

# Calculate total runs conceded (bat runs + extras)
deliveries['runs_conceded'] = deliveries['runs_off_bat'] + deliveries['extras']
runs_conceded = deliveries.groupby('bowler')['runs_conceded'].sum()

# Calculate balls bowled per bowler
balls_bowled = deliveries.groupby('bowler').size()

# Overs bowled
overs_bowled = balls_bowled / 6

# Economy rate
economy_rate = runs_conceded / overs_bowled
economy_df = economy_rate.reset_index(name='economy_rate')

# Merging wickets and economy
bowler_stats = pd.merge(bowler_wickets, economy_df, on='bowler')

# Top 10 bowlers by wickets
top_bowlers = bowler_stats.sort_values('wickets', ascending=False).head(10)

# Create bubble chart
fig = px.scatter(
    top_bowlers,
    x='economy_rate',
    y='wickets',
    size='wickets',
    color='economy_rate',
    color_continuous_scale='Viridis',
    hover_name='bowler',
    labels={'economy_rate': 'Economy Rate (Runs per Over)', 'wickets': 'Wickets Taken'},
    title='Top 10 Bowlers: Wickets vs Economy Rate'
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# Calculate total deliveries and dot balls per bowler
total_balls = deliveries.groupby('bowler').size()
dot_balls = deliveries[deliveries['runs_off_bat'] == 0].groupby('bowler').size()

dot_ball_percentage = (dot_balls / total_balls * 100).reset_index(name='DotBallPercent')
dot_ball_percentage['TotalBalls'] = total_balls.values

# Sort top 20 bowlers by dot ball percentage for better heatmap visualization
dot_ball_percentage = dot_ball_percentage.sort_values('DotBallPercent', ascending=False).head(20)

dot_ball_percentage_sorted = dot_ball_percentage.sort_values('DotBallPercent')

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dot_ball_percentage_sorted['DotBallPercent'],
    y=dot_ball_percentage_sorted['bowler'],
    mode='markers',
    marker=dict(color='purple', size=12), 
    name='Dot Ball %'
))
fig.add_trace(go.Scatter(
    x=dot_ball_percentage_sorted['DotBallPercent'],
    y=dot_ball_percentage_sorted['bowler'],
    mode='lines',
    line=dict(color='purple', width=2),
    showlegend=False
))

fig.update_layout(title='Top Bowlers Dot Ball Percentage - Lollipop Chart',
                  xaxis_title='Dot Ball Percentage (%)',
                  yaxis_title='Bowler',
                  yaxis=dict(autorange='reversed'))

st.plotly_chart(fig)
