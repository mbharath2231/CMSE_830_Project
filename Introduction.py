import streamlit as st

st.set_page_config(
    page_title="T20 World Cup Analytics",
    page_icon="🏏",
    layout="wide"
)

# 1. HEADER SECTION
st.title("🏏 T20 World Cup 2024: Data Science Portfolio")
st.markdown("### Fall 2025 Project")
st.divider()

# 2. PROJECT ABSTRACT (The "Why")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Project Objective")
    st.markdown("""
    This project applies **Machine Learning** and **Statistical Analysis** to T20 Cricket data to uncover winning strategies and predict match outcomes.
    
    Unlike standard dashboards that only look backward, this tool uses historical data to:
    * **Forecast** match winners using Random Forest algorithms.
    * **Cluster** players into performance roles (Anchors vs. Power Hitters) using K-Means.
    * **Analyze** head-to-head dominance and venue biases.
    """)
    
    st.info("**Data Source:** Ball-by-ball delivery data and match summaries from the ICC T20 World Cup.")

with col2:

    st.image(
        "india_win.webp",
        caption="Champions: Team India lifting the T20 World Cup 2024 Trophy",
        use_container_width=True
    )

# 3. KEY FEATURES (The "What")
st.divider()
st.subheader("🚀 Key Modules")

# Create 3 cards for your main features
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("### 🔮 Win Predictor")
    st.write("A **Supervised Learning** model (Random Forest) that predicts match outcomes based on Team Strength, Venue, and Toss decisions.")
    st.markdown("👉 *Go to: Match Predictor*")

with m2:
    st.markdown("### 🧩 Player Clustering")
    st.write("An **Unsupervised Learning** module (K-Means) that groups players based on Strike Rate, Average, and Economy to find hidden roles.")
    st.markdown("👉 *Go to: Player Roles*")

with m3:
    st.markdown("### 📊 Performance Deep Dive")
    st.write("Advanced EDA visualizations including Head-to-Head win rates, Venue Analysis, and custom MVP rankings.")
    st.markdown("👉 *Go to: Team Performance*")

# 4. TECH STACK (The "How")
st.divider()
st.markdown("### 🛠️ Tech Stack & Methodology")
st.markdown("""
- **Python**: Core Logic
- **Streamlit**: Web Interface
- **Scikit-Learn**: Machine Learning (Random Forest, K-Means)
- **Plotly**: Interactive Visualizations
- **Pandas/NumPy**: Data Manipulation & Preprocessing
""")

# 5. FOOTER
st.divider()
st.caption("Developed by **Bharath Mikkilineni, Caroline Newton, Hemprasanna Anbarasan** for Fall 2025. Michigan State University.")