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

Tabs:
- 👨‍🏏 Batsmen – strike rate, average, total runs  
- 🏹 Bowlers – wickets, economy, strike rate  
- 🏠 Teams – win %, toss performance, **opponent analysis**
"""
)

tab_bat, tab_bowl, tab_team = st.tabs(["👨‍🏏 Batsmen", "🏹 Bowlers", "🏠 Teams"])

# ============================================================
# TAB 1 – BATSMEN
# ============================================================
with tab_bat:
    # --------------------------------------------------------
    # Aggregate batting stats
    # --------------------------------------------------------
    batsman_stats = deliveries.groupby("striker").agg(
        total_runs=("runs_off_bat", "sum"),
        balls_faced=("ball", "count"),
        dismissals=("player_dismissed", lambda x: x.notna().sum()),
    ).reset_index()

    batsman_stats["strike_rate"] = (
        batsman_stats["total_runs"] / batsman_stats["balls_faced"] * 100
    ).round(2)

    batsman_stats["batting_average"] = batsman_stats.apply(
        lambda row: row["total_runs"] / row["dismissals"]
        if row["dismissals"] > 0 else None,
        axis=1,
    )
    batsman_stats["batting_average"] = batsman_stats["batting_average"].round(2)

    # ===================== Metrics Table =====================
    st.markdown("### 📋 Batsmen Metrics Table")
    st.markdown("""
**Purpose:** Show consolidated batting performance metrics (`total_runs`, `balls_faced`,
`strike_rate`, `batting_average`) for all batsmen who meet a minimum-balls filter.
""")

    min_balls = st.slider(
        "Minimum balls faced to include a batsman:",
        min_value=10, max_value=200, value=30, step=10,
    )
    filtered_batsman = batsman_stats[batsman_stats["balls_faced"] >= min_balls]

    st.dataframe(
        filtered_batsman.sort_values("total_runs", ascending=False).reset_index(drop=True)
    )

    st.markdown("""
**Data-driven conclusion:** - `total_runs` increases strongly with `balls_faced`, showing that longer stays at the crease drive scoring volume.  
- Batsmen with high `strike_rate` and few `dismissals` provide the most T20 impact.  
- `batting_average` is unstable for low-ball players, so filtering on `balls_faced` is necessary for reliable comparisons.  
""")

    # ===================== Top N Bar Chart =====================
    top_n = st.slider("Number of top batsmen to visualize:", 5, 30, 10, 1)

    metric_choice = st.selectbox(
        "Sort & rank batsmen by:",
        ["total_runs", "strike_rate", "batting_average"],
        index=0,
    )

    top_batsmen = (
        filtered_batsman.sort_values(metric_choice, ascending=False)
        .head(top_n)
    )

    st.markdown(f"### 📊 Top {top_n} Batsmen by `{metric_choice}`")
    st.markdown("""
**Purpose:** Rank the best-performing batsmen using a key metric to highlight leading scorers,
fastest hitters, or most consistent anchors.
""")

    fig_bat = px.bar(
        top_batsmen,
        x="striker",
        y=metric_choice,
        text=metric_choice,
        labels={"striker": "Batsman", metric_choice: metric_choice.replace("_", " ").title()},
    )
    fig_bat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bat, use_container_width=True)

    st.markdown(f"""
**Data-driven conclusion:** - Players at the top of `{metric_choice}` dominate that aspect of batting performance.  
- High `strike_rate` at the top of the chart identifies explosive scorers ideal for powerplay and death overs.  
- High `batting_average` combined with reasonable `strike_rate` points to stable anchors in the lineup.  
""")

    # ===================== 3D Performance Summary =====================
    st.markdown("### 🧭 3D Performance Summary – Top 10 Batsmen")
    st.markdown("""
**Purpose:** Compare top batsmen along three dimensions:
`total_runs`, `balls_faced`, and `dismissals`, with bubble size representing `strike_rate`.
""")

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
        size_max=24,
        hover_name="striker",
        title="Top 10 Batsmen: Runs vs Balls vs Dismissals",
        labels={
            "total_runs": "Total Runs",
            "balls_faced": "Balls Faced",
            "dismissals": "Dismissals"
        },
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("""
**Data-driven conclusion:** - `total_runs` rises sharply with `balls_faced`, confirming that volume hitters occupy the crease longer.  
- Batsmen with high `total_runs` and low `dismissals` are both productive and reliable.  
- Larger bubbles (higher `strike_rate`) highlight the rare players who are fast scorers **and** high scorers.  
""")

# ============================================================
# TAB 2 – BOWLERS
# ============================================================
with tab_bowl:
    # --------------------------------------------------------
    # Aggregate bowling stats
    # --------------------------------------------------------
    wickets_df = deliveries[deliveries["player_dismissed"].notna()]
    bowler_wickets = wickets_df.groupby("bowler").size().reset_index(name="wickets")

    deliveries["runs_conceded"] = deliveries["runs_off_bat"] + deliveries["extras"]
    runs_conceded = deliveries.groupby("bowler")["runs_conceded"].sum().reset_index()
    balls_bowled = deliveries.groupby("bowler").size().reset_index(name="balls_bowled")

    bowler_stats = (
        bowler_wickets.merge(runs_conceded, on="bowler", how="outer")
        .merge(balls_bowled, on="bowler", how="outer")
        .fillna(0)
    )

    bowler_stats["overs_bowled"] = bowler_stats["balls_bowled"] / 6

    bowler_stats["economy_rate"] = bowler_stats.apply(
        lambda row: row["runs_conceded"] / row["overs_bowled"]
        if row["overs_bowled"] > 0 else None,
        axis=1,
    ).round(2)

    bowler_stats["bowling_average"] = bowler_stats.apply(
        lambda row: row["runs_conceded"] / row["wickets"]
        if row["wickets"] > 0 else None,
        axis=1,
    ).round(2)

    bowler_stats["strike_rate"] = bowler_stats.apply(
        lambda row: row["balls_bowled"] / row["wickets"]
        if row["wickets"] > 0 else None,
        axis=1,
    ).round(2)

    # ===================== Bowler Table =====================
    st.markdown("### 📋 Bowler Metrics Table")
    st.markdown("""
**Purpose:** Summarize bowling performance using `wickets`, `runs_conceded`, `overs_bowled`,
`economy_rate`, `bowling_average`, and `strike_rate`.
""")

    min_overs = st.slider(
        "Minimum overs bowled to include a bowler:",
        min_value=1, max_value=20, value=4, step=1,
    )
    filtered_bowlers = bowler_stats[bowler_stats["overs_bowled"] >= min_overs]

    st.dataframe(
        filtered_bowlers.sort_values("wickets", ascending=False).reset_index(drop=True)
    )

    st.markdown("""
**Data-driven conclusion:** - Bowlers with high `wickets` and low `economy_rate` are the most valuable in T20.  
- `strike_rate` highlights wicket-takers who strike frequently.  
- Large `overs_bowled` combined with good metrics indicates trusted, frontline bowlers.  
""")

    # ===================== Bowler Scatter =====================
    st.markdown("### 📊 Wickets vs Economy Rate (Top Bowlers)")
    st.markdown("""
**Purpose:** Visualize the trade-off between wicket-taking (`wickets`) and run containment
(`economy_rate`), with bubble size showing `overs_bowled`.
""")

    top_n_bowl = st.slider("Number of top bowlers to visualize:", 5, 30, 10, 1)

    metric_choice_b = st.selectbox(
        "Select metric to rank bowlers:",
        ["wickets", "economy_rate", "bowling_average", "strike_rate"],
        index=0,
    )
    asc = metric_choice_b in ["economy_rate", "bowling_average", "strike_rate"]
    top_bowlers = (
        filtered_bowlers.sort_values(metric_choice_b, ascending=asc)
        .head(top_n_bowl)
    )

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

    st.markdown("""
**Data-driven conclusion:** - Elite bowlers appear in the region of **high `wickets` and low `economy_rate`**.  
- Large bubbles in that region show bowlers who handle both workload and impact.  
- Bowlers with many `wickets` but high `economy_rate` reflect aggressive, risk-heavy styles.  
""")

# ============================================================
# TAB 3 – TEAMS
# ============================================================
with tab_team:
    # --------------------------------------------------------
    # Aggregate team stats
    # --------------------------------------------------------
    teams = pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna())

    team_stats = []
    for team in teams:
        team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        matches_played = len(team_matches)
        wins = (team_matches["winner"] == team).sum()
        toss_wins = (team_matches["toss_winner"] == team).sum()

        team_stats.append(
            {
                "team": team,
                "matches_played": matches_played,
                "wins": wins,
                "win_pct": (wins / matches_played * 100) if matches_played else None,
                "toss_wins": toss_wins,
                "toss_win_pct": (toss_wins / matches_played * 100)
                if matches_played else None,
            }
        )

    team_stats_df = pd.DataFrame(team_stats).round(2)

    # ===================== Team Table =====================
    st.markdown("### 📋 Global Team Metrics")
    st.markdown("Overview of all teams' performance.")

    st.dataframe(
        team_stats_df.sort_values("win_pct", ascending=False).reset_index(drop=True)
    )

    # ===================== Global Charts =====================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Win % Leaderboard")
        fig_team = px.bar(
            team_stats_df.sort_values("win_pct", ascending=False),
            x="team",
            y="win_pct",
            text="win_pct",
            color="win_pct",
            color_continuous_scale="Viridis",
            labels={"team": "Team", "win_pct": "Win %"},
        )
        fig_team.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_team, use_container_width=True)

    with col2:
        st.markdown("#### 🎲 Toss Win %")
        fig_toss = px.bar(
            team_stats_df.sort_values("toss_win_pct", ascending=False),
            x="team",
            y="toss_win_pct",
            text="toss_win_pct",
            color="toss_win_pct",
            color_continuous_scale="Blues",
            labels={"team": "Team", "toss_win_pct": "Toss Win %"},
        )
        fig_toss.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_toss, use_container_width=True)

    # =========================================================
    # NEW SECTION: Specific Team Drill-Down
    # =========================================================
    st.divider()
    st.subheader("🕵️ Deep Dive: Specific Team Analysis")
    st.markdown("Select a team to see **who** they beat and **where** they win.")

    selected_team = st.selectbox("Select a Team:", sorted(teams))

    if selected_team:
        # 1. Filter matches where selected_team WON
        won_matches = matches[matches['winner'] == selected_team].copy()

        if won_matches.empty:
            st.warning(f"⚠️ {selected_team} has not won any matches in this dataset.")
        else:
            # 2. Identify Opponents in those wins
            # If team1 is winner, opponent is team2. If team2 is winner, opponent is team1.
            won_matches['opponent'] = won_matches.apply(
                lambda x: x['team2'] if x['team1'] == selected_team else x['team1'], axis=1
            )

            # 3. Aggregation
            wins_vs_opp = won_matches['opponent'].value_counts().reset_index()
            wins_vs_opp.columns = ['Opponent', 'Wins']

            wins_by_venue = won_matches['venue'].value_counts().reset_index()
            wins_by_venue.columns = ['Venue', 'Wins']

            # 4. Visualizations
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"#### 🆚 Who did {selected_team} beat?")
                fig_vs = px.bar(
                    wins_vs_opp,
                    x='Opponent',
                    y='Wins',
                    text='Wins',
                    color='Wins',
                    color_continuous_scale='Reds',
                    title=f"{selected_team} Wins vs. Opponents"
                )
                st.plotly_chart(fig_vs, use_container_width=True)

            with col_b:
                st.markdown(f"#### 🏟️ Where did {selected_team} win?")
                fig_venue = px.pie(
                    wins_by_venue,
                    names='Venue',
                    values='Wins',
                    title=f"{selected_team} Wins by Venue",
                    hole=0.4
                )
                st.plotly_chart(fig_venue, use_container_width=True)

            st.info(f"""
            **Analysis:**
            * **Dominance:** {selected_team} has the most wins against **{wins_vs_opp.iloc[0]['Opponent']}**.
            * **Fortress:** Their most successful venue is **{wins_by_venue.iloc[0]['Venue']}**.
            """)