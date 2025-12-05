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
This page runs **formal hypothesis tests** to check whether key cricket factors  
like `toss_winner` and `bat_first_team` have a statistically significant impact  
on the final `winner` of the match.
""")


# ============================================================
# TEST 1 — Toss Winner vs Match Winner
# ============================================================
st.header("🎲 Test 1: Does Winning the Toss Increase the Probability of Winning the Match?")
st.markdown("""
**Purpose:**  
Test whether teams that win the `toss_winner` are more likely to become the match `winner`.
""")

# Flags for toss winner == match winner / loser
matches["toss_win_and_match_win"] = matches.apply(
    lambda x: 1 if x["toss_winner"] == x["winner"] else 0, axis=1
)

matches["toss_win_and_match_loss"] = matches.apply(
    lambda x: 1 if x["toss_winner"] != x["winner"] else 0, axis=1
)

# Contingency table (1D: counts of win vs loss given toss_winner)
contingency_table = pd.DataFrame({
    "Match Won": [(matches["toss_winner"] == matches["winner"]).sum()],
    "Match Lost": [(matches["toss_winner"] != matches["winner"]).sum()],
})

st.markdown("### 📋 Contingency Table – `toss_winner` vs `winner`")
st.markdown("""
Shows how many times the team that won the toss also won or lost the match.
""")
st.dataframe(contingency_table)

# Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

st.markdown("### 📌 Hypothesis Setup")
st.markdown("""
- **H0 (Null):** `toss_winner` and `winner` are independent  
  → Winning the toss does **not** change the chance of winning the match.  
- **H1 (Alt):** `toss_winner` and `winner` are associated  
  → Winning the toss **does** change the chance of winning the match.
""")

st.write(f"**Chi-square statistic:** `{chi2:.4f}`")
st.write(f"**p-value:** `{p:.4f}`")

if p < 0.05:
    st.success("✔ Result is **statistically significant** (p < 0.05).")
else:
    st.info("✖ Result is **not statistically significant** (p ≥ 0.05).")

# Data-driven conclusion for Test 1
st.markdown("### ✅ Data-driven conclusion – Toss Impact")
if p < 0.05:
    st.markdown(f"""
- `toss_winner` and `winner` show a **significant statistical association** (p = `{p:.4f}` < 0.05).  
- Teams winning the `toss_winner` tend to win matches more often than expected by chance.  
- Toss decisions should be treated as a **non-trivial factor** in match outcome analysis.  
""")
else:
    st.markdown(f"""
- `toss_winner` and `winner` do **not** show a strong statistical association (p = `{p:.4f}` ≥ 0.05).  
- Teams winning the `toss_winner` do **not** win significantly more matches than expected by chance.  
- Match outcome appears to depend more on `team_strength`, `batting/bowling performance`, and in-game decisions than on the toss alone.  
""")


# ============================================================
# TEST 2 — Batting First vs Batting Second
# ============================================================
st.header("🏏 Test 2: Do Batting-First Teams Win More Often Than Batting-Second Teams?")
st.markdown("""
**Purpose:**  
Compare the win rates of `bat_first_team` and `field_first_team` to see  
whether batting first gives a measurable advantage.
""")

# Determine first batting team from toss decision
matches["bat_first_team"] = matches.apply(
    lambda x: x["toss_winner"]
    if x["toss_decision"] == "bat"
    else (x["team1"] if x["toss_winner"] != x["team1"] else x["team2"]),
    axis=1,
)

matches["field_first_team"] = matches.apply(
    lambda x: x["team1"] if x["bat_first_team"] != x["team1"] else x["team2"],
    axis=1,
)

# Win flags
matches["bat_first_win"] = matches["bat_first_team"] == matches["winner"]
matches["field_first_win"] = matches["field_first_team"] == matches["winner"]

# Counts for summary
bat_first_wins = matches["bat_first_win"].sum()
bat_first_total = len(matches)

field_first_wins = matches["field_first_win"].sum()
field_first_total = len(matches)

summary_df = pd.DataFrame({
    "Category": ["Bat First", "Field Second"],
    "Wins": [bat_first_wins, field_first_wins],
    "Total Matches": [bat_first_total, field_first_total],
    "Win %": [
        round(bat_first_wins / bat_first_total * 100, 2),
        round(field_first_wins / field_first_total * 100, 2),
    ],
})

st.markdown("### 📋 Summary Table – Bat First vs Field Second")
st.markdown("""
Shows how often teams win when they **bat first** compared to when they **chase**.
""")
st.dataframe(summary_df)

# Proportion Z-Test
counts = np.array([bat_first_wins, field_first_wins])
totals = np.array([bat_first_total, field_first_total])

stat, pval = proportions_ztest(counts, totals)

st.markdown("### 📌 Hypothesis Setup")
st.markdown("""
- **H0 (Null):** Win probability is the same for `bat_first_team` and `field_first_team`.  
- **H1 (Alt):** Win probability is **different** for `bat_first_team` compared to `field_first_team`.  
""")

st.write(f"**Z-statistic:** `{stat:.4f}`")
st.write(f"**p-value:** `{pval:.4f}`")

if pval < 0.05:
    st.success("✔ Difference in win rates is **statistically significant** (p < 0.05).")
else:
    st.info("✖ Difference in win rates is **not statistically significant** (p ≥ 0.05).")

# Data-driven conclusion for Test 2
st.markdown("### ✅ Data-driven conclusion – Batting Order Impact")
if pval < 0.05:
    st.markdown(f"""
- `bat_first_team` and `field_first_team` have **meaningfully different** win rates (p = `{pval:.4f}` < 0.05).  
- Either batting first or chasing offers a measurable advantage in this dataset, depending on which category has higher `Win %`.  
- Strategy around `toss_decision` (`bat` vs `field`) should be treated as a **key tactical choice**.  
""")
else:
    st.markdown(f"""
- Win rates for `bat_first_team` and `field_first_team` are **not statistically different** (p = `{pval:.4f}` ≥ 0.05).  
- Both batting first and chasing can be equally successful; there is **no clear universal advantage** from batting order alone.  
- Outcome seems driven more by execution (run rate, wickets, bowling performance) than by simply batting first or second.  
""")
