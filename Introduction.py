import streamlit as st

st.set_page_config(page_title="T20 WorldCup Dashboard", layout="wide")

# Import Google Fonts and custom styles with animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(130deg, #0f2027, #203a43, #2c5364);
    margin: 0;
    padding: 0;
}

.title {
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(45deg, #F4D03F, #F39C12, #D35400);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 3s infinite alternate;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.intro {
    max-width: 700px;
    color: #d8d8d8;
    font-size: 1.3rem;
    text-align: center;
    margin: 0 auto 3rem auto;
    padding: 1.5rem;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.1);
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(4px);
    transition: transform 0.3s ease;
}

.intro:hover {
    transform: scale(1.05);
}

.img-container {
    display: flex;
    justify-content: center;
    margin-bottom: 4rem;
}

.img-container img {
    width: 75%;
    max-width: 900px;
    border-radius: 25px;
    box-shadow: 0 20px 50px rgba(255, 192, 64, 0.6);
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}

.img-container img:hover {
    transform: scale(1.05);
    box-shadow: 0 25px 75px rgba(255, 192, 64, 0.9);
}

.caption {
    text-align: center;
    color: #f0e68c;
    font-style: italic;
    font-size: 1.1rem;
    margin-top: 0.5rem;
    filter: drop-shadow(0 0 2px #d4af37);
}
</style>

<div class="title">ICC T20 WorldCup Dashboard</div>

<div class="intro">
Welcome to the interactive <strong>T20 World Cup 2024</strong> analytics app.<br>
Use the sidebar to navigate through different pages — each page lets you explore a specific aspect of the tournament.
</div>

<div class="img-container">
    <img src="https://c.ndtvimg.com/2023-12/qhgfbe4o_t20-world-cup-2024-logo_625x300_07_December_23.jpg?output-quality=80&downsize=1200:*" alt="T20 World Cup Logo" />
</div>

<div class="caption">Visualize, analyze, and explore!</div>
""", unsafe_allow_html=True)
