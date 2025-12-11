import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Match Prediction", layout="wide")
st.title("🔮 Match Winner Prediction & Model Tuning")

st.markdown("""
### 📋 Project Module: Predictive Analytics & Tuning
This module compares three algorithms to predict match winners. 
Use the **Sidebar** to tune model hyperparameters and view **Cross-Validation** results.

**The Models:**
1.  **Random Forest:** Non-linear ensemble. Best for complex patterns.
2.  **Logistic Regression:** Probabilistic. Best for inference (Odds Ratios).
3.  **Linear Regression:** Baseline. (Not recommended for classification).
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
    # 3. SIDEBAR: HYPERPARAMETER TUNING
    st.sidebar.header("⚙️ Hyperparameter Tuning")
    st.sidebar.markdown("Adjust model settings to optimize performance.")

    # Random Forest Params
    st.sidebar.subheader("🌲 Random Forest")
    rf_n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100, step=10, 
                                      help="More trees = more stable predictions, but slower.")
    rf_max_depth = st.sidebar.slider("Max Depth", 1, 20, 10, 
                                   help="Controls how complex the trees can get. Deeper trees capture more detail but may overfit.")

    # Logistic Regression Params
    st.sidebar.subheader("📈 Logistic Regression")
    log_C = st.sidebar.select_slider("Regularization Strength (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0,
                                   help="Inverse of regularization strength. Smaller values specify stronger regularization.")

    # Cross Validation Settings
    st.sidebar.divider()
    st.sidebar.subheader("🔁 Cross Validation")
    cv_folds = st.sidebar.slider("CV Folds (k)", 2, 10, 5, help="Number of times to split the data and re-test.")

    # 4. DATA SPLIT
    X = df[['team1_code', 'team2_code', 'venue_code', 'team1_toss_win', 'toss_bat']]
    y = df['team1_win']
    feature_names = ['Team 1 ID', 'Team 2 ID', 'Venue', 'Toss Won', 'Bat First']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. MODEL TRAINING & CV
    # Random Forest
    rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_cv = cross_val_score(rf, X, y, cv=cv_folds).mean() # Cross Validation Score

    # Logistic Regression
    log = LogisticRegression(C=log_C, max_iter=1000)
    log.fit(X_train, y_train)
    log_pred = log.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)
    log_cv = cross_val_score(log, X, y, cv=cv_folds).mean()

    # Linear Regression (Baseline - No tuning needed for simple OLS)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred_raw = lin.predict(X_test)
    lin_pred = [1 if x > 0.5 else 0 for x in lin_pred_raw]
    lin_acc = accuracy_score(y_test, lin_pred)
    # Lin reg CV isn't standard for classification accuracy comparison here, skipping to focus on classifiers.

    # 6. METRICS DASHBOARD
    st.divider()
    st.subheader("📊 Model Performance (Test Set vs Cross-Validation)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest (Test)", f"{rf_acc:.1%}", f"CV Mean: {rf_cv:.1%}", delta_color="normal")
    col2.metric("Logistic Regression", f"{log_acc:.1%}", f"CV Mean: {log_cv:.1%}", delta_color="normal")
    col3.metric("Linear Regression", f"{lin_acc:.1%}", "Baseline (No CV)", delta_color="off")

    # 7. DETAILED ANALYSIS TABS
    st.divider()
    st.subheader("🔍 Deep Dive: Inside the Black Box")
    
    tab1, tab2, tab3 = st.tabs(["🌲 Random Forest", "📈 Logistic Regression", "📏 Linear Regression"])

    # --- TAB 1: RANDOM FOREST ---
    with tab1:
        st.write("### 1. Random Forest Classifier")
        st.markdown(f"""
        **Current Hyperparameters:** `n_estimators={rf_n_estimators}`, `max_depth={rf_max_depth}`.
        * **Cross-Validation Score:** {rf_cv:.1%} (Average of {cv_folds} runs).
        
        """)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            cm_rf = confusion_matrix(y_test, rf_pred)
            fig_cm = ff.create_annotated_heatmap(cm_rf, x=['Pred Loss', 'Pred Win'], y=['Actual Loss', 'Actual Win'], colorscale='Greens')
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c2:
            st.write("**Feature Importance**")
            importances = rf.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 2: LOGISTIC REGRESSION ---
    with tab2:
        st.write("### 2. Logistic Regression")
        st.markdown(f"""
        **Current Hyperparameters:** `C={log_C}` (Regularization Strength).
        * **Cross-Validation Score:** {log_cv:.1%} (Average of {cv_folds} runs).
        """)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            cm_log = confusion_matrix(y_test, log_pred)
            fig_cm_log = ff.create_annotated_heatmap(cm_log, x=['Pred Loss', 'Pred Win'], y=['Actual Loss', 'Actual Win'], colorscale='Blues')
            fig_cm_log.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm_log, use_container_width=True)
            
        with c2:
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
        st.markdown("Linear regression is used here only as a baseline to demonstrate why classification algorithms are preferred for binary outcomes.")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            res_df = pd.DataFrame({'Actual': y_test, 'Predicted': lin_pred_raw})
            fig_res = px.scatter(res_df, x='Actual', y='Predicted', title="Residuals", opacity=0.4)
            fig_res.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
        with c2:
            st.error("Issues:")
            st.write(f"- **R-Squared:** {r2_score(y_test, lin_pred_raw):.3f}")
            st.write("- **Out of Bounds:** Predictions can exceed [0, 1]")

    # 8. INTERACTIVE PREDICTOR
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
        vec = pd.DataFrame([{
            'team1_code': le_team.transform([t1])[0],
            'team2_code': le_team.transform([t2])[0],
            'venue_code': le_venue.transform([venue])[0],
            'team1_toss_win': 1 if toss == t1 else 0,
            'toss_bat': 1 if dec == "Bat" else 0
        }])
        
        rf_p = rf.predict_proba(vec)[0][1]
        log_p = log.predict_proba(vec)[0][1]
        lin_p = lin.predict(vec)[0]
        
        st.write("---")
        # FIXED: Use consistent variable names (res1, res2, res3)
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
            
        with res3:  # This variable name was the issue (was res_3)
            st.markdown("#### 📏 Linear Reg")
            winner = t1 if lin_p > 0.5 else t2
            st.warning(f"**{winner}**")
            st.write(f"Raw Output: {lin_p:.3f}")

    # 9. DOCUMENTATION
    st.divider()
    with st.expander("📖 Concept Guide: Tuning & Validation"):
        st.markdown("""
        ### **1. Hyperparameter Tuning**
        Models have "settings" that aren't learned from data, called **Hyperparameters**.
        * **Random Forest:** Changing `n_estimators` (trees) or `max_depth` changes how complex the model is.
        * **Logistic Regression:** Changing `C` controls regularization (preventing overfitting).
        * **Goal:** Use the sidebar to find the settings that give the highest Test Accuracy.

        ### **2. Cross-Validation (CV)**
        * **Problem:** A single Test Accuracy might be lucky (or unlucky) based on how we split the data.
        * **Solution:** CV splits the data into **k parts** (folds). It trains on k-1 and tests on 1, repeating this k times.
        * **Result:** The "CV Mean" is a much more reliable measure of how the model will perform on new data.
        """)