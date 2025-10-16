import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Top Batsmen in the Tournament')

# Load deliveries data
deliveries = pd.read_csv('data/deliveries.csv')

# Top 10 batsmen by total runs
batsman_runs = deliveries.groupby('striker')['runs_off_bat'].sum().reset_index()
top_batsmen = batsman_runs.sort_values('runs_off_bat', ascending=False).head(10)

fig_batsmen = px.bar(
    top_batsmen,
    y='striker',
    x='runs_off_bat',
    orientation='h',
    color='runs_off_bat',
    color_continuous_scale='Viridis',
    labels={'striker': 'Batsman', 'runs_off_bat': 'Total Runs'},
    title='Top 10 Batsmen by Runs',
    hover_data={'runs_off_bat': ':,', 'striker': True}
)
fig_batsmen.update_layout(yaxis={'categoryorder':'total ascending'})

st.plotly_chart(fig_batsmen, use_container_width=True)

# # Top 10 bowlers by wickets taken
# wickets = deliveries[deliveries['player_dismissed'].notna()]
# bowler_wickets = wickets.groupby('bowler').size().reset_index(name='wickets')
# top_bowlers = bowler_wickets.sort_values('wickets', ascending=False).head(10)

# fig_bowlers = px.bar(
#     top_bowlers,
#     y='bowler',
#     x='wickets',
#     orientation='h',
#     color='wickets',
#     color_continuous_scale='Plasma',
#     labels={'bowler': 'Bowler', 'wickets': 'Wickets Taken'},
#     title='Top 10 Bowlers by Wickets',
#     hover_data={'wickets': ':,', 'bowler': True}
# )
# fig_bowlers.update_layout(yaxis={'categoryorder':'total ascending'})

# st.plotly_chart(fig_bowlers, use_container_width=True)

# Filter top batsmen by total runs

st.title("Top 10 Batsmen Performance Summary")

# Load deliveries data
deliveries = pd.read_csv('data/deliveries.csv')

# Calculate key stats for each batsman
batsman_stats = deliveries.groupby('striker').agg(
    total_runs=('runs_off_bat', 'sum'),
    balls_faced=('ball', 'count'),
    dismissals=('player_dismissed', lambda x: x.notnull().sum())
).reset_index()

# Calculate strike rate
batsman_stats['strike_rate'] = (batsman_stats['total_runs'] / batsman_stats['balls_faced']) * 100

# Filter top 10 batsmen by total runs
top_batsmen = batsman_stats.nlargest(10, 'total_runs')

# Create 3D scatter plot
fig = px.scatter_3d(
    top_batsmen,
    x='total_runs',
    y='balls_faced',
    z='dismissals',
    color='striker',
    size='strike_rate',
    size_max=25,
    hover_name='striker',
    title='Top 10 Batsmen: Runs vs Balls Faced vs Dismissals',
    labels={
        'total_runs': 'Total Runs',
        'balls_faced': 'Balls Faced',
        'dismissals': 'Dismissals'
    }
)

# Adjust layout for a larger plot
fig.update_layout(
    scene=dict(
        xaxis_title='Total Runs',
        yaxis_title='Balls Faced',
        zaxis_title='Dismissals'
    ),
    autosize=False,
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=40)
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=False)