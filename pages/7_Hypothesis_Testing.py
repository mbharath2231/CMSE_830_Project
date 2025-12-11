import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# 1. PAGE CONFIG & CUSTOM CSS
st.set_page_config(page_title="Hypothesis Testing", layout="wide")

st.markdown("""
<style>
    /* Card Styling */
    .stat-card {
        background-color: #f8f9fa;
        border-left: 5px solid #636EFA;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-title { font-size: 1.1rem; color: #555; text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { font-size: 2.5rem; font-weight: bold; color: #333; }
    .stat-note { font-size: 0.9rem; color: #777; }
    
    /* Result Box Styling */
    .result-box-pass { background-color: #d4edda; border: 2px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 10px; text-align: center; }
    .result-box-fail { background-color: #f8d7da; border: 2px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Statistical Inference & Theory")
st.markdown("""
**Goal:** Apply core statistical theorems to real-world sports data.
We validate patterns using **Hypothesis Testing** and demonstrate the **Central Limit Theorem**.
""")

# 2. LOAD DATA
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("data/matches.csv")
        deliveries = pd.read_csv("data/deliveries.csv")
        return matches, deliveries
    except FileNotFoundError:
        st.error("❌ Data files not found.")
        return None, None

matches, deliveries = load_data()

if matches is not None and deliveries is not None:
    # PRE-PROCESSING
    matches = matches.dropna(subset=['toss_winner', 'winner', 'toss_decision'])
    matches = matches[matches['winner'] != 'No Result']

    # TABS
    tab1, tab2, tab3 = st.tabs(["🎲 Test 1: The Toss", "🏏 Test 2: Bat vs Chase", "📉 Theory: Central Limit Theorem"])

    # ==============================================================================
    # TEST 1: TOSS EFFECT (CHI-SQUARE)
    # ==============================================================================
    with tab1:
        st.subheader("1. Does Winning the Toss help you Win the Match?")
        
        matches['toss_outcome'] = matches.apply(
            lambda x: "Won Match" if x['toss_winner'] == x['winner'] else "Lost Match", axis=1
        )
        
        toss_wins = matches[matches['toss_outcome'] == "Won Match"].shape[0]
        total_matches = matches.shape[0]
        toss_win_rate = toss_wins / total_matches
        
        observed = [toss_wins, total_matches - toss_wins]
        expected = [total_matches/2, total_matches/2] 
        chi2, p = chi2_contingency([observed, expected])[0:2]
        
        col_viz, col_stat = st.columns([1.5, 1])
        
        with col_viz:
            fig_toss = px.pie(
                matches, names='toss_outcome', 
                title="Outcome for Teams that Won the Toss",
                color='toss_outcome',
                color_discrete_map={'Won Match': '#00CC96', 'Lost Match': '#EF553B'},
                hole=0.6
            )
            fig_toss.update_traces(textinfo='percent+label', textfont_size=16)
            st.plotly_chart(fig_toss, use_container_width=True)
            
        with col_stat:
            st.markdown("### 📊 Test Statistics")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">P-Value</div>
                <div class="stat-value">{p:.4f}</div>
                <div class="stat-note">Threshold: 0.05</div>
            </div>
            """, unsafe_allow_html=True)
            
            if p < 0.05:
                st.markdown('<div class="result-box-pass">✅ RESULT: SIGNIFICANT</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box-fail">⚖️ RESULT: NOT SIGNIFICANT</div>', unsafe_allow_html=True)

    # ==============================================================================
    # TEST 2: BAT FIRST vs CHASE (Z-TEST)
    # ==============================================================================
    with tab2:
        st.subheader("2. Is there an advantage to Batting First?")
        
        def get_strategy_winner(row):
            toss_win_team = row['toss_winner']
            decision = row['toss_decision']
            winner = row['winner']
            if decision == 'bat':
                bat_first_team = toss_win_team
            else:
                bat_first_team = row['team2'] if toss_win_team == row['team1'] else row['team1']
            return "Bat First" if bat_first_team == winner else "Chase"

        matches['strategy_outcome'] = matches.apply(get_strategy_winner, axis=1)
        counts = matches['strategy_outcome'].value_counts()
        n_bat_first = counts.get("Bat First", 0)
        n_chase = counts.get("Chase", 0)
        total = n_bat_first + n_chase
        
        stat, pval = proportions_ztest(n_bat_first, total, value=0.5)
        ci_low, ci_high = proportion_confint(n_bat_first, total, alpha=0.05, method='wilson')
        
        col_viz2, col_stat2 = st.columns([1.5, 1])
        
        with col_viz2:
            plot_df = pd.DataFrame({
                'Strategy': ['Bat First', 'Chase'],
                'Win Rate': [n_bat_first/total, n_chase/total]
            })
            fig_bar = px.bar(
                plot_df, x='Strategy', y='Win Rate', color='Strategy',
                text_auto='.1%', title="Win Rates by Strategy",
                color_discrete_map={'Bat First': '#636EFA', 'Chase': '#AB63FA'},
                range_y=[0, 1]
            )
            fig_bar.add_hline(y=0.5, line_dash="dash", line_color="black", annotation_text="50% (Neutral)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_stat2:
            st.markdown("### 📊 Test Statistics")
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #AB63FA;">
                <div class="stat-title">P-Value</div>
                <div class="stat-value">{pval:.4f}</div>
            </div>
            <div class="stat-card" style="border-left-color: #AB63FA;">
                <div class="stat-title">95% Confidence Interval</div>
                <div class="stat-value" style="font-size: 1.8rem;">{ci_low:.1%} — {ci_high:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    # ==============================================================================
    # TAB 3: CENTRAL LIMIT THEOREM (NEW FEATURE)
    # ==============================================================================
    with tab3:
        st.subheader("📉 The Central Limit Theorem (CLT) in Action")
        st.markdown("""
        **The Problem:** Individual batsman scores are NOT Normal (Bell Curve). They are skewed (lots of 0s and 10s).
        **The Solution:** CLT states that the **Average** of many samples *will* form a Bell Curve. 
        This is why we can use Z-Tests and T-Tests on sports data!
        """)

        # 1. Prepare Population Data (Individual Scores)
        if 'runs_off_bat' in deliveries.columns:
            # Aggregate runs per innings per batsman
            batter_scores = deliveries.groupby(['match_id', 'striker'])['runs_off_bat'].sum().reset_index()
            population = batter_scores['runs_off_bat'].values
        else:
            st.error("Deliveries data missing 'runs_off_bat'.")
            st.stop()

        col_clt1, col_clt2 = st.columns(2)

        with col_clt1:
            st.markdown("### 1. The Population (Reality)")
            fig_pop = px.histogram(population, nbins=50, title="Distribution of Individual Scores", 
                                 labels={'value': 'Runs Scored'}, color_discrete_sequence=['#EF553B'])
            fig_pop.update_layout(showlegend=False)
            st.plotly_chart(fig_pop, use_container_width=True)
            st.warning("⚠️ **Highly Skewed:** Most scores are low. This is NOT a Bell Curve.")

        with col_clt2:
            st.markdown("### 2. The Sampling Distribution (CLT)")
            
            # Interactive Slider
            sample_size = st.slider("Sample Size (Matches per Average)", min_value=2, max_value=100, value=30)
            
            # Run Simulation
            sample_means = []
            for _ in range(1000): # 1000 trials
                # Randomly pick 'sample_size' innings and take the mean
                sample = np.random.choice(population, size=sample_size, replace=True)
                sample_means.append(sample.mean())
            
            fig_clt = px.histogram(sample_means, nbins=50, title=f"Distribution of Sample Means (n={sample_size})",
                                 labels={'value': 'Average Runs'}, color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_clt, use_container_width=True)
            st.success("✅ **Normal Distribution:** Even though individual scores are skewed, their *Averages* form a perfect Bell Curve!")

    # ==============================================================================
    # DOCUMENTATION
    # ==============================================================================
    st.divider()
    with st.expander("📖 Concept Guide - How to Read This"):
        st.markdown("""
        ### **1. The P-Value**
        * Think of the P-Value as a **"Luck Meter"**.
        * **If P < 0.05:** It means the result is **Real**. (Luck is unlikely).
        
        ### **2. Central Limit Theorem (CLT)**
        * It is the reason Statistics works.
        * Try moving the slider in Tab 3. Notice how the messy red graph turns into a smooth green Bell Curve as you increase the sample size? That is CLT.
        """)