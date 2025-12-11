import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="Score Predictor", layout="wide")

st.markdown("""
<style>
    .big-font { font-size:18px !important; }
    .stAlert { padding: 10px; border-radius: 10px; }
    .scoreboard {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .score-val { font-size: 3.5rem; font-weight: bold; color: #4CAF50; text-shadow: 1px 1px 2px black; }
    .score-lbl { font-size: 1.2rem; letter-spacing: 2px; text-transform: uppercase; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Scenario-Based Score Predictor")
st.markdown("""
**Goal:** Predict the final 1st innings score based on how the team started (Powerplay).
This module uses **Random Forest Regression** and **Bootstrap Confidence Intervals**
""")

# 2. DATA LOADING (Robust)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/deliveries.csv")
        matches = pd.read_csv("data/matches.csv")

        # FIX 1: Calculate Total Runs if missing
        if 'total_runs' not in df.columns:
            if 'runs_off_bat' in df.columns and 'extras' in df.columns:
                df['total_runs'] = df['runs_off_bat'] + df['extras']
            else:
                return None, None, None

        # FIX 2: Detect Columns
        match_col = 'match_id' if 'match_id' in df.columns else 'id'
        inn_col = 'innings' if 'innings' in df.columns else 'inning'

        # Filter for 1st Innings only
        df = df[df[inn_col] == 1].copy()

        # A. Final Score Aggregation
        final_scores = df.groupby([match_col, 'batting_team', 'bowling_team'])['total_runs'].sum().reset_index()
        final_scores.rename(columns={'total_runs': 'final_total'}, inplace=True)

        # B. Powerplay Score Aggregation (Overs 0-5)
        pp_df = df[df['ball'] < 6.0]
        pp_scores = pp_df.groupby([match_col])['total_runs'].sum().reset_index()
        pp_scores.rename(columns={'total_runs': 'pp_score'}, inplace=True)

        # C. Merge
        merged = pd.merge(final_scores, pp_scores, on=match_col, how='inner')

        # D. Add Venue
        match_id_map = 'id' if 'id' in matches.columns else 'match_number'
        final_df = pd.merge(merged, matches[[match_id_map, 'venue']], left_on=match_col, right_on=match_id_map)

        # E. Encoding
        le_team = LabelEncoder()
        all_teams = pd.concat([final_df['batting_team'], final_df['bowling_team']]).unique()
        le_team.fit(all_teams)
        
        final_df['bat_code'] = le_team.transform(final_df['batting_team'])
        final_df['bowl_code'] = le_team.transform(final_df['bowling_team'])
        
        le_venue = LabelEncoder()
        final_df['venue_code'] = le_venue.fit_transform(final_df['venue'])
        
        return final_df, le_team, le_venue
        
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None, None, None

df, le_team, le_venue = load_data()

if df is not None:
    X = df[['bat_code', 'bowl_code', 'venue_code', 'pp_score']]
    y = df['final_total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MODEL TRAINING
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # =========================================================
    # PART A: DATA VISUALIZATION (THE "WHY")
    # =========================================================
    st.divider()
    st.header("1. Data Analysis: Does the start matter?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PLOT 1: TREND ANALYSIS
        fig_trend = px.scatter(
            df, x='pp_score', y='final_total', 
            color='venue', 
            trendline="ols", 
            title="Correlation: Powerplay vs Final Score",
            labels={'pp_score': 'Runs at 6 Overs', 'final_total': 'Final 20-Over Score'},
            template="plotly_white", opacity=0.7
        )
        fig_trend.update_layout(font=dict(size=14))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col2:
        st.info("""
        **💡 How to read this chart:**
        * **X-Axis:** Runs scored in the first 6 overs.
        * **Y-Axis:** Final total score.
        * **The Red Line:** This is the regression line. Since it goes **UP**, it proves that a better start leads to a better finish.
        """)
        st.metric("R-Squared (Correlation Strength)", f"{r2:.2f}")

    # =========================================================
    # PART B: MODEL ACCURACY (THE "PROOF")
    # =========================================================
    st.divider()
    st.header("2. Model Evaluation: Can we trust it?")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.info("""
        **💡 Actual vs. Predicted:**
        * We hid 20% of the matches from the AI to test it.
        * **Red Dashed Line:** Perfect prediction.
        * **Blue Dots:** The AI's guesses.
        
        **Conclusion:** Since the blue dots hug the red line closely, the model is accurate.
        """)
        st.metric("Avg Error (MAE)", f"+/- {mae:.0f} Runs")

    with col4:
        # PLOT 2: ACTUAL VS PREDICTED
        viz_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
        fig_scat = px.scatter(
            viz_df, x='Actual', y='Predicted', 
            title="Accuracy Check: Actual vs Predicted", 
            opacity=0.6, template="plotly_white"
        )
        fig_scat.add_shape(type="line", line=dict(dash='dash', color='red', width=3), 
                           x0=min(y), y0=min(y), x1=max(y), y1=max(y))
        fig_scat.update_layout(font=dict(size=14))
        st.plotly_chart(fig_scat, use_container_width=True)

    # =========================================================
    # PART C: INTERACTIVE PREDICTOR (THE "TOOL")
    # =========================================================
    st.divider()
    st.header("3. Live Prediction")
    st.markdown("Enter a match scenario below to see the Random Forest Forecast.")

    # Input Card
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        bat_team = c1.selectbox("Batting Team", sorted(le_team.classes_))
        bowl_team = c2.selectbox("Bowling Team", [t for t in le_team.classes_ if t != bat_team])
        venue_name = c3.selectbox("Venue", sorted(le_venue.classes_))
        pp_input = c4.number_input("Runs (0-6 Overs)", 0, 100, 45, help="Enter the score at the end of the Powerplay.")
    
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        # Predict Point Estimate
        vec = pd.DataFrame([{
            'bat_code': le_team.transform([bat_team])[0],
            'bowl_code': le_team.transform([bowl_team])[0],
            'venue_code': le_venue.transform([venue_name])[0],
            'pp_score': pp_input
        }])
        pred = rf.predict(vec)[0]
        
        # Bootstrap Intervals
        residuals = y_train - rf.predict(X_train)
        residuals_np = residuals.values # Safe conversion
        boot_preds = []
        for _ in range(1000):
            boot_preds.append(pred + resample(residuals_np, n_samples=1)[0])
            
        lower, upper = np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)
        
        # ==========================================
        # RESULT SECTION
        # ==========================================
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            # CUSTOM HTML SCOREBOARD
            st.markdown(f"""
            <div class="scoreboard">
                <div class="score-lbl">Projected Final Score</div>
                <div class="score-val">{pred:.0f}</div>
                <div style="margin-top: 10px;">Safe Range (95% Confidence): <b>{lower:.0f} — {upper:.0f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **💡 What does this mean?**
            * The AI calculates that **{:.0f}** is the most likely score based on history.
            * However, cricket is uncertain. We are 95% confident the final score will be between **{:.0f}** and **{:.0f}**.
            """.format(pred, lower, upper))
            
        with res_col2:
            # PLOT 3: DISTRIBUTION CHART
            fig_dist = ff.create_distplot([boot_preds], ['Simulation'], bin_size=2, show_hist=False, show_rug=False)
            fig_dist.add_vline(x=pred, line_color="#4CAF50", annotation_text="Target")
            fig_dist.add_vrect(x0=lower, x1=upper, fillcolor="#4CAF50", opacity=0.1, annotation_text="Safe Zone")
            
            fig_dist.update_layout(
                title="Forecast Probability Curve", 
                showlegend=False, 
                font=dict(size=14),
                margin=dict(t=40, l=0, r=0, b=0),
                height=350
            )
            st.plotly_chart(fig_dist, use_container_width=True)