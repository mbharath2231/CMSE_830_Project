import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🏏 Performance Metrics Dashboard")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

st.markdown(
    """
This page summarizes **player** and **team** performance using cricket-specific metrics.

Use the tabs below to explore:

- 👨‍🏏 **Batsman metrics** (strike rate, average, total runs)
- 🏹 **Bowler metrics** (wickets, economy, strike rate)
- 🏠 **Team metrics** (win %, toss impact)
"""
)

tab_bat, tab_bowl, tab_team = st.tabs(["👨‍🏏 Batsmen", "🏹 Bowlers", "🏠 Teams"])

# ============================================
# TAB 1 – BATSMEN
# ============================================
with tab_bat:
    st.subheader("👨‍🏏 Batsman Performance Metrics")

    # Aggregate batting stats
    batsman_stats = deliveries.groupby("striker").agg(
        total_runs=("runs_off_bat", "sum"),
        balls_faced=("ball", "count"),
        dismissals=("player_dismissed", lambda x: x.notna().sum()),
    ).reset_index()

    # Derived metrics
    batsman_stats["strike_rate"] = (
        batsman_stats["total_runs"] / batsman_stats["balls_faced"] * 100
    ).round(2)

    # Avoid division by zero in average
    batsman_stats["batting_average"] = batsman_stats.apply(
        lambda row: row["total_runs"] / row["dismissals"]
        if row["dismissals"] > 0
        else None,
        axis=1,
    )
    batsman_stats["batting_average"] = batsman_stats["batting_average"].round(2)

    # Minimum balls filter to avoid noise
    min_balls = st.slider(
        "Minimum balls faced to include a batsman:",
        min_value=10,
        max_value=200,
        value=30,
        step=10,
    )
    filtered_batsman = batsman_stats[batsman_stats["balls_faced"] >= min_balls]

    st.markdown("#### 📋 Metrics Table")
    st.dataframe(
        filtered_batsman.sort_values("total_runs", ascending=False).reset_index(drop=True)
    )

    # Top N selector
    top_n = st.slider("Number of top batsmen to visualize:", 5, 30, 10, 1)

    metric_choice = st.selectbox(
        "Sort by metric:",
        ["total_runs", "strike_rate", "batting_average"],
        index=0,
    )

    top_batsmen = (
        filtered_batsman.sort_values(metric_choice, ascending=False)
        .head(top_n)
    )

    st.markdown(f"#### 📊 Top {top_n} Batsmen by `{metric_choice}`")

    fig_bat = px.bar(
        top_batsmen,
        x="striker",
        y=metric_choice,
        text=metric_choice,
        labels={"striker": "Batsman", metric_choice: metric_choice.replace("_", " ").title()},
    )
    fig_bat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bat, use_container_width=True)

        # --------------------------------------------
    # 3D Performance Summary for Top 10 Batsmen
    # --------------------------------------------
    st.markdown("#### 🧭 3D Performance Summary – Top 10 Batsmen")

    top10_for_3d = batsman_stats.nlargest(10, "total_runs").copy()
    top10_for_3d["strike_rate"] = (
        top10_for_3d["total_runs"] / top10_for_3d["balls_faced"] * 100
    )

    fig3d = px.scatter_3d(
        top10_for_3d,
        x="total_runs",
        y="balls_faced",
        z="dismissals",
        color="striker",
        size="strike_rate",
        size_max=25,
        hover_name="striker",
        title="Top 10 Batsmen: Runs vs Balls Faced vs Dismissals",
        labels={
            "total_runs": "Total Runs",
            "balls_faced": "Balls Faced",
            "dismissals": "Dismissals"
        },
    )

    fig3d.update_layout(
        scene=dict(
            xaxis_title='Total Runs',
            yaxis_title='Balls Faced',
            zaxis_title='Dismissals'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig3d, use_container_width=True)


# ============================================
# TAB 2 – BOWLERS
# ============================================
with tab_bowl:
    st.subheader("🏹 Bowler Performance Metrics")

    # Wickets: any dismissal credited to bowler
    wickets_df = deliveries[deliveries["player_dismissed"].notna()]
    bowler_wickets = (
        wickets_df.groupby("bowler")
        .size()
        .reset_index(name="wickets")
    )

    # Runs conceded: bat runs + extras
    deliveries["runs_conceded"] = deliveries["runs_off_bat"] + deliveries["extras"]
    runs_conceded = deliveries.groupby("bowler")["runs_conceded"].sum().reset_index()

    # Balls bowled
    balls_bowled = deliveries.groupby("bowler").size().reset_index(name="balls_bowled")

    # Merge all
    bowler_stats = (
        bowler_wickets
        .merge(runs_conceded, on="bowler", how="outer")
        .merge(balls_bowled, on="bowler", how="outer")
        .fillna(0)
    )

    # Derived metrics
    bowler_stats["overs_bowled"] = bowler_stats["balls_bowled"] / 6
    bowler_stats["economy_rate"] = bowler_stats.apply(
        lambda row: row["runs_conceded"] / row["overs_bowled"]
        if row["overs_bowled"] > 0
        else None,
        axis=1,
    )
    bowler_stats["economy_rate"] = bowler_stats["economy_rate"].round(2)

    bowler_stats["bowling_average"] = bowler_stats.apply(
        lambda row: row["runs_conceded"] / row["wickets"]
        if row["wickets"] > 0
        else None,
        axis=1,
    )
    bowler_stats["bowling_average"] = bowler_stats["bowling_average"].round(2)

    bowler_stats["strike_rate"] = bowler_stats.apply(
        lambda row: row["balls_bowled"] / row["wickets"]
        if row["wickets"] > 0
        else None,
        axis=1,
    )
    bowler_stats["strike_rate"] = bowler_stats["strike_rate"].round(2)

    # Filter out part-time bowlers
    min_overs = st.slider(
        "Minimum overs bowled to include a bowler:",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
    )
    filtered_bowlers = bowler_stats[bowler_stats["overs_bowled"] >= min_overs]

    st.markdown("#### 📋 Metrics Table")
    st.dataframe(
        filtered_bowlers.sort_values("wickets", ascending=False).reset_index(drop=True)
    )

    top_n_bowl = st.slider("Number of top bowlers to visualize:", 5, 30, 10, 1)

    metric_choice_b = st.selectbox(
        "Sort by metric:",
        ["wickets", "economy_rate", "bowling_average", "strike_rate"],
        index=0,
    )

    asc = metric_choice_b in ["economy_rate", "bowling_average", "strike_rate"]
    top_bowlers = (
        filtered_bowlers.sort_values(metric_choice_b, ascending=asc)
        .head(top_n_bowl)
    )

    st.markdown(f"#### 📊 Top {top_n_bowl} Bowlers by `{metric_choice_b}`")

    fig_bowl = px.scatter(
        top_bowlers,
        x="economy_rate",
        y="wickets",
        size="overs_bowled",
        hover_name="bowler",
        labels={
            "economy_rate": "Economy Rate (Runs / Over)",
            "wickets": "Wickets",
            "overs_bowled": "Overs Bowled",
        },
    )
    st.plotly_chart(fig_bowl, use_container_width=True)

# ============================================
# TAB 3 – TEAMS
# ============================================
with tab_team:
    st.subheader("🏠 Team Performance Metrics")

    # Get all teams from team1/team2
    teams = pd.unique(
        pd.concat([matches["team1"], matches["team2"]], ignore_index=True)
    )
    teams = pd.Series(teams).dropna().unique()

    # Build team metrics
    team_stats = []

    for team in teams:
        team_matches = matches[
            (matches["team1"] == team) | (matches["team2"] == team)
        ]
        matches_played = len(team_matches)

        wins = (team_matches["winner"] == team).sum()

        toss_wins = 0
        if "toss_winner" in matches.columns:
            toss_wins = (team_matches["toss_winner"] == team).sum()

        team_stats.append(
            {
                "team": team,
                "matches_played": matches_played,
                "wins": wins,
                "win_pct": (wins / matches_played * 100) if matches_played > 0 else None,
                "toss_wins": toss_wins,
                "toss_win_pct": (toss_wins / matches_played * 100)
                if matches_played > 0
                else None,
            }
        )

    team_stats_df = pd.DataFrame(team_stats)
    team_stats_df["win_pct"] = team_stats_df["win_pct"].round(2)
    team_stats_df["toss_win_pct"] = team_stats_df["toss_win_pct"].round(2)

    st.markdown("#### 📋 Team Metrics Table")
    st.dataframe(
        team_stats_df.sort_values("win_pct", ascending=False).reset_index(drop=True)
    )

    st.markdown("#### 📊 Team Win Percentage")

    fig_team = px.bar(
        team_stats_df.sort_values("win_pct", ascending=False),
        x="team",
        y="win_pct",
        text="win_pct",
        labels={"team": "Team", "win_pct": "Win %"},
    )
    fig_team.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_team, use_container_width=True)

    if "toss_winner" in matches.columns:
        st.markdown("#### 🎲 Toss Win Percentage by Team")

        fig_toss = px.bar(
            team_stats_df.sort_values("toss_win_pct", ascending=False),
            x="team",
            y="toss_win_pct",
            text="toss_win_pct",
            labels={"team": "Team", "toss_win_pct": "Toss Win %"},
        )
        fig_toss.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_toss, use_container_width=True)
    else:
        st.info(
            "Column `toss_winner` not found in matches dataset – toss-based metrics skipped."
        )
