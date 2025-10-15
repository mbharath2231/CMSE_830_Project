🏏 ICC T20 World Cup 2024 Data Analysis & Visualization
📘 Overview

This project performs Exploratory Data Analysis (EDA) and Interactive Visualization on the ICC T20 World Cup 2024 dataset.
The goal is to explore patterns in team performance, player consistency, match outcomes, and scoring trends using data-driven insights.

The project was developed as part of my coursework for CMSE 830 — to apply statistical, probabilistic, and visualization concepts through a real-world dataset.

🧩 Why I Chose This Dataset

I chose the ICC T20 World Cup 2024 dataset because:

Cricket is a data-rich sport that offers multiple perspectives (players, teams, matches, seasons).
It provides a perfect opportunity to apply statistical methods like probability, hypothesis testing, and regression.
The dataset includes multiple files (matches.csv, deliveries.csv), which allowed me to practice data merging, cleaning, and integration — an essential real-world skill.
The data naturally supports EDA, pattern discovery, and interactive visualization, aligning perfectly with the project requirements.

What I Learned from IDA & EDA

During the Initial Data Analysis (IDA) and Exploratory Data Analysis (EDA), I learned to:

Handle and understand missing data, and decide when to impute or drop it.
Perform univariate, bivariate, and multivariate analysis to understand relationships between features.
Conduct correlation analysis and identify the strongest variables affecting match outcomes.
Explore patterns and trends in cricket statistics — such as toss effects, team dominance, and seasonal scoring.

Formulate testable hypotheses like:
Does winning the toss increase the probability of winning the match?
Has the scoring trend increased over the seasons?

Streamlit App: Interactive Visualization

I developed a Streamlit web app for interactive exploration of the T20 dataset.

💻 Key Features:

Filter matches by season
View top winning teams
Analyze average runs per season
Visualize toss decision trends
Compare batsmen and bowlers performances
All visualizations implemented using Seaborn & Matplotlib

Do batting-first teams win more often?

EDA helped me connect statistical reasoning with real sports behavior, bridging data science with analytical storytelling.

To Run Locally : 
pip install streamlit pandas seaborn matplotlib
streamlit run app.py
