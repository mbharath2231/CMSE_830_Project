import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Team Performance Dashboard")

matches = pd.read_csv('data/matches.csv')

teams = sorted(matches['winner'].dropna().unique())
selected_team = st.selectbox("Select a team to view detailed performance:", options=["All"] + teams)

if selected_team == "All":
    # Show total wins for all teams
    wins = matches['winner'].value_counts().reset_index()
    wins.columns = ['Team', 'Wins']

    fig = px.bar(
        wins,
        x='Team',
        y='Wins',
        color='Wins',
        text='Wins',
        title='🏆 Total Wins by Team',
        color_continuous_scale='Tealgrn'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    # Show performance vs opponents for selected team
    team_matches = matches[(matches['team1'] == selected_team) | (matches['team2'] == selected_team)]
    team_matches['opponent'] = team_matches.apply(lambda x: x['team2'] if x['team1'] == selected_team else x['team1'], axis=1)

    summary = team_matches.groupby('opponent').agg(
        Wins=('winner', lambda x: (x == selected_team).sum()),
        Matches=('opponent', 'count')
    ).reset_index()

    fig = px.scatter(
        summary,
        x='opponent',
        y='Wins',
        size='Matches',
        color='Wins',
        color_continuous_scale='Viridis',
        title=f"{selected_team}: Performance vs Each Opponent",
        hover_name='opponent'
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

    venue_wins = team_matches[team_matches['winner'] == selected_team]
    venue_summary = venue_wins['venue'].value_counts().reset_index()
    venue_summary.columns = ['Venue', 'Wins']

    fig2 = px.bar(
        venue_summary,
        x='Venue',
        y='Wins',
        color='Wins',
        text='Wins',
        title=f"{selected_team}: Wins by Venue",
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig2, use_container_width=True)
