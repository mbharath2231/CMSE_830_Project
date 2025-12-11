import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Match Prediction", layout="wide")
st.title("🔮 Match Winner Prediction & Model Showdown")

st.markdown("""
### 📋 Project Module: Predictive Analytics
We compare three distinct algorithms to solve the classification problem: **"Who will win the match?"**

**The Models:**
1.  **Random Forest (Ensemble):** A non-linear model that builds multiple decision trees. Best for capturing complex interactions.
2.  **Logistic Regression (GLM):** A probabilistic model that estimates log-odds. Best for statistical inference.
3.  **Linear Regression (LPM):** A baseline Linear Probability Model. Included to demonstrate why regression is often poor for classification.
""")

# 2. DATA LOADING & PREP
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/matches.csv')
        df = df.dropna(subset=['winner', 'toss_winner', 'toss_decision', 'venue'])
        df = df[df['winner'] != 'No Result']
        
        # Feature Engineering
        model_df = df.copy()
        model_df['team1_win'] = (model_df['winner'] == model_df['team1']).astype(int)
        model_df['team1_toss_win'] = (model_df['toss_winner'] == model_df['team1']).astype(int)
        model_df['toss_bat'] = model_df['toss_decision'].apply(lambda x: 1 if x == 'bat' else 0)
        
        # Encoders
        le_venue = LabelEncoder()
        model_df['venue_code'] = le_venue.fit_transform(model_df['venue'])
        
        le_team = LabelEncoder()
        all_teams = pd.concat([model_df['team1'], model_df['team2']]).unique()
        le_team.fit(all_teams)
        model_df['team1_code'] = le_team.transform(model_df['team1'])
        model_df['team2_code'] = le_team.transform(model_df['team2'])
        
        return model_df, le_team, le_venue
    except FileNotFoundError:
        st.error("❌ Data not found!")
        return None, None, None

df, le_team, le_venue = load_data()

if df is not None:
    # We rely on raw Team IDs (Categorical). Random Forest handles this better than Linear Models.
    X = df[['team1_code', 'team2_code', 'venue_code', 'team1_toss_win', 'toss_bat']]
    y = df['team1_win']
    
    # Meaningful names for visualizations
    feature_names = ['Team 1 ID', 'Team 2 ID', 'Venue', 'Toss Won', 'Bat First']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MODEL TRAINING
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    # Logistic Regression
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    log_pred = log.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)

    # Linear Regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred_raw = lin.predict(X_test)
    lin_pred = [1 if x > 0.5 else 0 for x in lin_pred_raw] # Threshold
    lin_acc = accuracy_score(y_test, lin_pred)

    # 5. DETAILED MODEL ANALYSIS (TABS)
    st.divider()
    st.subheader("🔍 Deep Dive: Inside the Black Box")
    
    tab1, tab2, tab3 = st.tabs(["🌲 Random Forest", "📈 Logistic Regression", "📏 Linear Regression"])

    # --- TAB 1: RANDOM FOREST ---
    with tab1:
        st.write("### 1. Random Forest Classifier")
        st.markdown("""
        **Theory:** Random Forest is an ensemble method. It creates 100+ "weak" decision trees and averages their votes.
        * **Pros:** Can handle non-linear logic (e.g., "Team A is good, BUT only at Home").
        * **Cons:** Harder to interpret mathematically than regression.
        """)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            # Confusion Matrix
            cm_rf = confusion_matrix(y_test, rf_pred)
            fig_cm = ff.create_annotated_heatmap(cm_rf, x=['Pred Loss', 'Pred Win'], y=['Actual Loss', 'Actual Win'], colorscale='Greens')
            fig_cm.update_layout(title="Confusion Matrix (Random Forest)")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c2:
            # Feature Importance
            st.write("**What matters most?**")
            importances = rf.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 2: LOGISTIC REGRESSION ---
    with tab2:
        st.write("### 2. Logistic Regression (Inference)")
        st.markdown("""
        **Theory:** Fits a Sigmoid curve to estimate probability ($P$).
        $$ P(y=1) = \\frac{1}{1 + e^{-(b_0 + b_1x)}} $$
        **Allows us to calculate **Odds Ratios**.
        """)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            # Confusion Matrix
            cm_log = confusion_matrix(y_test, log_pred)
            fig_cm_log = ff.create_annotated_heatmap(cm_log, x=['Pred Loss', 'Pred Win'], y=['Actual Loss', 'Actual Win'], colorscale='Blues')
            fig_cm_log.update_layout(title="Confusion Matrix (Logistic)")
            st.plotly_chart(fig_cm_log, use_container_width=True)
            
        with c2:
            # Odds Ratios
            st.write("**Odds Ratios**")
            coefs = log.coef_[0]
            odds = np.exp(coefs)
            coef_df = pd.DataFrame({'Feature': feature_names, 'Odds Ratio': odds})
            fig_odds = px.bar(coef_df, x='Odds Ratio', y='Feature', title="Odds Ratios (>1 = Good)")
            fig_odds.add_vline(x=1, line_dash="dash", line_color="red")
            st.plotly_chart(fig_odds, use_container_width=True)

    # --- TAB 3: LINEAR REGRESSION ---
    with tab3:
        st.write("### 3. Linear Regression (Baseline)")
        st.markdown("""
        **Theory:** Fits a straight line.
        * **Flaw:** It can predict probabilities > 100% or < 0%.
        * **Use Case:** We use this to show *why* classifiers are better for this task.
        """)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            # Residual Plot
            res_df = pd.DataFrame({'Actual': y_test, 'Predicted': lin_pred_raw})
            fig_res = px.scatter(res_df, x='Actual', y='Predicted', title="Residuals: Predictions are noisy", opacity=0.4)
            fig_res.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
        with c2:
            st.error("Linear Model Issues:")
            st.write(f"- **R-Squared:** {r2_score(y_test, lin_pred_raw):.3f} (Low is bad)")
            st.write("- **Out of Bounds:** Predictions can exceed [0, 1]")

    # 6. INTERACTIVE PREDICTOR
    st.divider()
    st.subheader("🚀 Live Prediction Simulation")
    st.markdown("Select a scenario below. We will feed it to **all three models** simultaneously.")

    col_in1, col_in2, col_in3 = st.columns(3)
    t1 = col_in1.selectbox("Team 1", sorted(le_team.classes_))
    t2_opts = [t for t in sorted(le_team.classes_) if t != t1]
    t2 = col_in2.selectbox("Team 2", t2_opts)
    venue = col_in3.selectbox("Venue", sorted(le_venue.classes_))
    
    col_in4, col_in5 = st.columns(2)
    toss = col_in4.selectbox("Toss Winner", [t1, t2])
    dec = col_in5.selectbox("Toss Decision", ["Bat", "Field"])
    
    if st.button("Run Models", type="primary"):
        # Input Vector
        vec = pd.DataFrame([{
            'team1_code': le_team.transform([t1])[0],
            'team2_code': le_team.transform([t2])[0],
            'venue_code': le_venue.transform([venue])[0],
            'team1_toss_win': 1 if toss == t1 else 0,
            'toss_bat': 1 if dec == "Bat" else 0
        }])
        
        # Get Predictions
        rf_p = rf.predict_proba(vec)[0][1]
        log_p = log.predict_proba(vec)[0][1]
        lin_p = lin.predict(vec)[0] # Raw output
        
        # Display Cards
        st.write("---")
        res1, res2, res3 = st.columns(3)
        
        with res1:
            st.markdown("#### 🌲 Random Forest")
            winner = t1 if rf_p > 0.5 else t2
            st.success(f"**{winner}**")
            st.progress(rf_p if winner==t1 else 1-rf_p, text=f"Confidence: {max(rf_p, 1-rf_p):.1%}")
            
        with res2:
            st.markdown("#### 📈 Logistic Reg")
            winner = t1 if log_p > 0.5 else t2
            st.info(f"**{winner}**")
            st.progress(log_p if winner==t1 else 1-log_p, text=f"Probability: {max(log_p, 1-log_p):.1%}")
            
        with res3:
            st.markdown("#### 📏 Linear Reg")
            winner = t1 if lin_p > 0.5 else t2
            st.warning(f"**{winner}**")
            st.write(f"Raw Output: {lin_p:.3f}")
            st.caption("Note: Value can be outside 0-1 range.")
            
    # 7. FINAL VERDICT
    st.divider()
    st.info(f"""
    ### 🏆 Official Recommendation
    Since we are predicting a binary outcome (Win/Loss) using categorical data (Team Names), **Random Forest** ({rf_acc:.1%}) is the scientifically correct choice. 
    
    Linear Regression ({lin_acc:.1%}) performs worse because it treats Team IDs as math numbers (e.g., Team 10 is 'bigger' than Team 5), which is logically flawed.
    """)