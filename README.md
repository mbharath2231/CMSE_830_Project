# 🏏 T20 World Cup 2024: Analytics & Prediction Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

> **Course:** CSE 482 (Big Data Analysis)  
> **Institution:** Michigan State University  
> **Semester:** Fall 2025

## 📖 Project Overview
This project is a full-stack Data Science application designed to analyze, visualize, and predict outcomes for the ICC T20 World Cup. Unlike traditional sports dashboards that only display historical statistics, this tool integrates **Machine Learning** to provide predictive insights.

The system utilizes **Supervised Learning (Random Forest)** to forecast match winners and **Unsupervised Learning (K-Means Clustering)** to identify player roles automatically based on performance metrics.

---

## 🚀 Key Features

### 1. 🔮 Match Outcome Predictor (Supervised Learning)
- **Algorithm:** Random Forest Classifier (`n_estimators=100`).
- **Function:** Predicts the winner between two teams based on venue, toss decision, and historical head-to-head strength.
- **Output:** Returns a probability confidence score (e.g., "India has a 64.2% chance of winning").

### 2. 🧩 Player Role Clustering (Unsupervised Learning)
- **Algorithm:** K-Means Clustering.
- **Function:** dynamically groups players into categories (e.g., "Anchors," "Power Hitters," "Economical Bowlers") based on Strike Rate, Average, and Economy.
- **Innovation:** Moves beyond standard positions (Batsman/Bowler) to reveal actual playing styles.

### 3. 🧹 Data Cleaning & Imputation
- Includes a dedicated module for handling missing values (`NaN`).
- Supports multiple imputation strategies: **Mean, Median, Mode, and Forward Fill**.
- Visualizes the impact of imputation on data distribution in real-time.

### 4. 📊 Advanced EDA (Exploratory Data Analysis)
- **Head-to-Head Analysis:** Drill-down view of specific team rivalries.
- **Venue Fortress:** Identifies which stadiums favor specific teams.
- **Correlation Heatmaps:** Analyzes relationships between variables (e.g., Run Rate vs. Win Probability).

---

## 🛠️ Tech Stack

| Component | Technology | Usage |
|-----------|------------|-------|
| **Language** | Python 3.x | Core logic and data manipulation |
| **Frontend** | Streamlit | Interactive web interface and dashboarding |
| **ML Models** | Scikit-Learn | Random Forest (Prediction), K-Means (Clustering) |
| **Data Processing** | Pandas, NumPy | Data cleaning, aggregation, and feature engineering |
| **Visualization** | Plotly Express | Interactive charts (3D Scatter, Heatmaps, Bar Charts) |

---

## 📂 Project Structure

```text
T20-WorldCup-Analytics/
├── data/
│   ├── matches.csv          # Match-level summary data
│   └── deliveries.csv       # Ball-by-ball delivery data
├── pages/
│   ├── 1_Datasets.py            # Data Inventory & Audit
│   ├── 3_Correlation.py         # Statistical Correlation
│   ├── 4_Imputation.py          # Data Cleaning Module
│   ├── 6_Team_Performance.py    # Drill-down Team Analytics
│   ├── 14_Players_Clustering.py # K-Means Clustering (Batsmen/Bowlers)
│   └── match_predictor.py       # Random Forest Win Predictor
├── Introduction.py          # Home/Landing Page (Main Entry Point)
├── requirements.txt         # Python dependencies
└── README.md                # Project Documentation