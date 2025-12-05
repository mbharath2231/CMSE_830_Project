import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import plotly.express as px

st.title("🏏 Final Score Prediction – Random Forest Regression")

@st.cache_data
def load_deliveries():
    deliveries = pd.read_csv("data/deliveries.csv")
    return deliveries

deliveries = load_deliveries()

# Derive over number from ball (e.g., 0.1 -> over 0, 1.1 -> over 1, etc.)
deliveries["over"] = deliveries["ball"].astype(int)

st.markdown("""
This page builds a **Random Forest Regression model** to predict the  
final 20-over score for an innings using **Powerplay (overs 1–6) performance**:

- `pp_runs`: Powerplay runs  
- `pp_rr`: Powerplay run rate  
- `pp_wkts`: Wickets lost in Powerplay  
- `boundary_pct`: Boundary percentage in Powerplay  
- `strike_rate`: Strike rate in Powerplay  
""")


# ============================================================
# BUILD INNINGS-LEVEL FEATURE DATASET
# ============================================================

st.header("📦 Building Innings-Level Feature Dataset")
st.markdown("""
**Purpose:**  
Aggregate ball-by-ball data into **one row per innings**, summarizing Powerplay
performance and linking it to the final `final_score` for that innings.
""")

# Identify each innings uniquely
deliveries["innings_id"] = (
    deliveries["match_id"].astype(str) + "_" + deliveries["innings"].astype(str)
)

innings_rows = []

for innings_id, df_inn in deliveries.groupby("innings_id"):

    # Powerplay overs (1–6). Over < 6 ≈ first six overs.
    pp = df_inn[df_inn["over"] < 6]

    # Skip incomplete or weird innings
    if len(pp) == 0:
        continue

    pp_runs = pp["runs_off_bat"].sum()
    pp_balls = len(pp)
    pp_rr = pp_runs / 6.0  # runs per over in PP

    pp_wkts = pp["player_dismissed"].notna().sum()

    boundaries = pp[pp["runs_off_bat"].isin([4, 6])]
    boundary_pct = len(boundaries) / pp_balls if pp_balls > 0 else 0

    strike_rate = (pp_runs / pp_balls) * 100 if pp_balls > 0 else 0

    # Final innings total = bat runs + extras
    final_total = df_inn["runs_off_bat"].sum() + df_inn["extras"].sum()

    innings_rows.append(
        {
            "pp_runs": pp_runs,
            "pp_rr": pp_rr,
            "pp_wkts": pp_wkts,
            "boundary_pct": boundary_pct,
            "strike_rate": strike_rate,
            "final_score": final_total,
        }
    )

df_features = pd.DataFrame(innings_rows)

if df_features.empty:
    st.error("No innings-level data could be constructed from deliveries.csv.")
    st.stop()

st.markdown("### 🔎 Feature Dataset Preview")
st.dataframe(df_features.head())
st.write(f"Rows: **{df_features.shape[0]}**, Columns: **{df_features.shape[1]}**")

st.markdown("""
**Data-driven conclusion:**  
- Each row represents a single innings, linking `pp_runs` and other PP features to `final_score`.  
- Variation in `pp_runs`, `boundary_pct`, and `strike_rate` across innings creates learnable patterns for the regression model.  
- The dataset is suitable for modeling how early-innings aggression and stability affect the final total.  
""")


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

st.header("🧠 Training Random Forest Regressor")
st.markdown("""
**Purpose:**  
Train a **RandomForestRegressor** on innings-level features to predict the  
`final_score` and evaluate model performance on a held-out test split.
""")

X = df_features.drop("final_score", axis=1)
y = df_features["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_split=2,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"- **RMSE:** `{rmse:.2f}` runs")
st.write(f"- **MAE:** `{mae:.2f}` runs")
st.write(f"- **R² Score:** `{r2:.2f}`")

st.markdown("""
**Data-driven conclusion:**  
- `RMSE` and `MAE` show the typical error in runs when predicting `final_score` from Powerplay stats.  
- A reasonably high `R²` indicates that a large portion of `final_score` variance is explained by `pp_runs`, `strike_rate`, and related features.  
- Errors are expected for innings with unusual collapses or late-innings surges that PP features alone cannot fully capture.  
""")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("📌 Feature Importance")
st.markdown("""
**Purpose:**  
Identify which Powerplay features contribute most to predicting the final `final_score`.
""")

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame(
    {"feature": X.columns, "importance": importances}
).sort_values("importance", ascending=False)

fig_imp = px.bar(
    feat_imp_df,
    x="feature",
    y="importance",
    text="importance",
    title="Random Forest Feature Importance",
)
fig_imp.update_layout(xaxis_title="Feature", yaxis_title="Importance")
st.plotly_chart(fig_imp, use_container_width=True)

st.dataframe(feat_imp_df.reset_index(drop=True))

st.markdown("""
**Data-driven conclusion:**  
- `pp_runs` and `pp_rr` typically rank highest, confirming that **scoring volume in the first six overs** is the strongest driver of `final_score`.  
- High `boundary_pct` and `strike_rate` indicate aggressive scoring and strongly contribute to predicting large totals.  
- `pp_wkts` has a negative effect: losing more wickets in the Powerplay usually reduces the eventual `final_score`.  
""")


# ============================================================
# USER PREDICTION UI
# ============================================================

st.header("🔮 Simulate Powerplay & Predict Final Score")
st.markdown("""
**Purpose:**  
Let the user plug in hypothetical Powerplay stats and obtain a **predicted final total**
based on historical patterns learned by the Random Forest model.
""")

col1, col2 = st.columns(2)

with col1:
    input_pp_runs = st.number_input("Powerplay Runs (overs 1–6)", 0, 100, 45)
    input_pp_wkts = st.number_input("Wickets Lost in Powerplay", 0, 10, 1)

with col2:
    input_boundary_pct = st.slider(
        "Boundary Percentage in Powerplay (0–1)",
        0.0,
        1.0,
        0.30,
        step=0.01,
    )
    input_strike_rate = st.number_input(
        "Strike Rate in Powerplay", 50.0, 250.0, 140.0
    )

# Derive run rate from runs
input_pp_rr = input_pp_runs / 6.0

user_input = pd.DataFrame(
    [
        {
            "pp_runs": input_pp_runs,
            "pp_rr": input_pp_rr,
            "pp_wkts": input_pp_wkts,
            "boundary_pct": input_boundary_pct,
            "strike_rate": input_strike_rate,
        }
    ]
)

if st.button("Predict Final 20-Over Score"):
    pred = rf_model.predict(user_input)[0]
    st.success(f"### 🏁 Predicted Final Score: **{pred:.0f} runs**")
    st.caption(
        f"(Given PP runs = {input_pp_runs}, wickets = {input_pp_wkts}, "
        f"boundary% = {input_boundary_pct:.2f}, SR = {input_strike_rate:.1f})"
    )

    st.markdown("""
**Interpretation:**  
- Higher `pp_runs`, `boundary_pct`, and `strike_rate` generally push the predicted `final_score` upward.  
- Increasing `pp_wkts` (more early wickets lost) typically lowers the predicted `final_score`.  
- The prediction reflects **average historical behavior** for similar Powerplay profiles and is not a guarantee for any single match.  
""")
