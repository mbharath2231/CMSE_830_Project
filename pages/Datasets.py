import streamlit as st
import pandas as pd

st.set_page_config(page_title='Datasets Display', layout='wide')

# Custom styled header
st.markdown("<h1 style='text-align: center; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif;'>Datasets Overview</h1>", unsafe_allow_html=True)

# Load datasets
matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

# Display datasets in expandable sections
with st.expander('Matches Dataset'):
    st.dataframe(matches)

with st.expander('Deliveries Dataset'):
    st.dataframe(deliveries)
