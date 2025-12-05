import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

st.title("📊 Statistical Analysis & Hypothesis Testing")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    return matches

matches = load_data()

st.markdown("""
This page performs **formal statistical hypothesis tests** to understand  
whether certain cricket factors significantly affect match outcomes.
""")

# ============================================================
# TEST 1 — Does Winning the Toss Increase Win Probability?
# ============================================================

st.header("🎲 Test 1: Does Winning the Toss Increase the Probability of Winning the Match?")

# Create columns for contingency table
matches["toss_win_and_match_win"] = matches.apply(
    lambda x: 1 if x["toss_winner"] == x["winner"] else 0, axis=1
)

matches["toss_win_and_match_loss"] = matches.apply(
    lambda x: 1 if x["toss_winner"] != x["winner"] else 0, axis=1
)

# Contingency table
contingency_table = pd.DataFrame({
    "Match Won": [ (matches["toss_winner"] == matches["winner"]).sum() ],
    "Match Lost": [ (matches["toss_winner"] != matches["winner"]).sum() ]
})

st.write("### 📋 Contingency Table (Toss Winner vs Match Outcome)")
st.dataframe(contingency_table)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

st.write("### 📌 Hypothesis")
st.write("""
- **H0 (Null)**: Winning the toss has *no effect* on winning the match.  
- **H1 (Alt)**: Winning the toss *does affect* winning the match.
""")

st.write(f"**Chi-square Statistic:** {chi2:.4f}")
st.write(f"**p-value:** {p:.4f}")

if p < 0.05:
    st.success("✔ The result is **statistically significant** (p < 0.05). Winning the toss **does influence** the chance of winning.")
else:
    st.info("✖ The result is **NOT statistically significant**. Winning the toss **does NOT meaningfully impact** the chance of winning.")

# ============================================================
# TEST 2 — Do Batting-First Teams Win More Often?
# ============================================================

st.header("🏏 Test 2: Do Batting-First Teams Win More Often Than Batting-Second Teams?")

# Determine first batting team
matches["bat_first_team"] = matches.apply(
    lambda x: x["toss_winner"] if x["toss_decision"] == "bat" else (x["team1"] if x["toss_winner"] != x["team1"] else x["team2"]),
    axis=1
)

matches["field_first_team"] = matches.apply(
    lambda x: x["team1"] if x["bat_first_team"] != x["team1"] else x["team2"],
    axis=1
)

# Win flags
matches["bat_first_win"] = matches["bat_first_team"] == matches["winner"]
matches["field_first_win"] = matches["field_first_team"] == matches["winner"]

# Counts
bat_first_wins = matches["bat_first_win"].sum()
bat_first_total = len(matches)

field_first_wins = matches["field_first_win"].sum()
field_first_total = len(matches)

st.write("### 📋 Summary Table")
summary_df = pd.DataFrame({
    "Category": ["Bat First", "Field First"],
    "Wins": [bat_first_wins, field_first_wins],
    "Total Matches": [bat_first_total, field_first_total],
    "Win %": [
        round(bat_first_wins / bat_first_total * 100, 2),
        round(field_first_wins / field_first_total * 100, 2)
    ]
})
st.dataframe(summary_df)

# Proportion Z-Test
counts = np.array([bat_first_wins, field_first_wins])
totals = np.array([bat_first_total, field_first_total])

stat, pval = proportions_ztest(counts, totals)

st.write("### 📌 Hypothesis")
st.write("""
- **H0 (Null):** Batting first and batting second have *equal chances* of winning.  
- **H1 (Alt):** Batting first gives a *different* (higher or lower) chance of winning.
""")

st.write(f"**Z-statistic:** {stat:.4f}")
st.write(f"**p-value:** {pval:.4f}")

if pval < 0.05:
    st.success("✔ The difference in win rates is **statistically significant** (p < 0.05). Batting first *does* impact the chance of winning.")
else:
    st.info("✖ The difference is **not statistically significant**. Batting first does *not* offer a clear win advantage.")
