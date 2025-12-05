import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

st.title("🔮 Match Winner Prediction")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    return matches

matches = load_data()

st.markdown("""
This model predicts the **probability of Team 1 winning the match** using:
- Toss winner  
- Toss decision  
- Venue  
- Team strength (historical win rate)
""")

# ================================================
# BUILD FEATURES
# ================================================

# Compute team win percentages
team_stats = matches["winner"].value_counts() / (
    matches["team1"].value_counts() + matches["team2"].value_counts()
)
team_win_rate = team_stats.fillna(0).to_dict()

# Create model dataset
df = matches.copy()

# Remove missing winners (if any)
df = df[df["winner"].notna()]

# ---------------------------
# Feature: team win strength
# ---------------------------
df["team1_strength"] = df["team1"].apply(lambda x: team_win_rate.get(x, 0))
df["team2_strength"] = df["team2"].apply(lambda x: team_win_rate.get(x, 0))

# ---------------------------
# Feature: toss winner binary
# ---------------------------
df["toss_winner_flag"] = (df["toss_winner"] == df["team1"]).astype(int)

# ---------------------------
# Feature: toss decision
# ---------------------------
df["toss_decision_flag"] = df["toss_decision"].apply(lambda x: 1 if x == "bat" else 0)

# ---------------------------
# Target variable
# ---------------------------
df["winner_flag"] = (df["winner"] == df["team1"]).astype(int)

# ---------------------------
# Final feature set
# ---------------------------
X = df[
    ["team1", "team2", "venue", "team1_strength", "team2_strength",
     "toss_winner_flag", "toss_decision_flag"]
]

y = df["winner_flag"]

# One-hot encode categorical features
cat_features = ["team1", "team2", "venue"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="passthrough"
)

# Build pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit the model
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
st.write("**Confusion Matrix:**")
st.write(cm)

st.markdown("---")

# ================================================
# USER PREDICTION INPUT
# ================================================
st.header("🧮 Predict a Match Outcome")

col1, col2 = st.columns(2)

teams = sorted(pd.unique(matches["team1"].dropna().unique().tolist() + matches["team2"].dropna().unique().tolist()))
venues = sorted(matches["venue"].dropna().unique())

with col1:
    team1_input = st.selectbox("Select Team 1", teams)
    team2_input = st.selectbox("Select Team 2", teams)

with col2:
    venue_input = st.selectbox("Select Venue", venues)
    toss_winner_input = st.selectbox("Who won the toss?", [team1_input, team2_input])
    toss_decision_input = st.selectbox("Toss decision", ["bat", "field"])

# Prepare input row
def prepare_input():
    return pd.DataFrame([{
        "team1": team1_input,
        "team2": team2_input,
        "venue": venue_input,
        "team1_strength": team_win_rate.get(team1_input, 0),
        "team2_strength": team_win_rate.get(team2_input, 0),
        "toss_winner_flag": 1 if toss_winner_input == team1_input else 0,
        "toss_decision_flag": 1 if toss_decision_input == "bat" else 0,
    }])

# Predict button
if st.button("Predict Winner"):
    input_df = prepare_input()
    prob = model.predict_proba(input_df)[0][1]  # probability team1 wins
    st.write(f"### 🔮 Probability **{team1_input}** Wins: **{prob*100:.2f}%**")

    if prob > 0.5:
        st.success(f"Model predicts **{team1_input}** is more likely to win.")
    else:
        st.info(f"Model predicts **{team2_input}** is more likely to win.")
