import joblib
import streamlit as st
import numpy as np

from classifier import config

st.beta_set_page_config(page_title="Nederlandse Sentiment Classificatie",
                        page_icon="https://raw.githubusercontent.com/maxscheijen/maxscheijen.github.io/main/favicon.ico")

# Load model
clf = joblib.load(config.MODEL_NAME)

# Header
st.markdown("<h1>Sentiment Classificatie</h1>", unsafe_allow_html=True)

# Get user input
sentence = str(st.text_input(label='Voer hier uw text in:'))

# Predict input user
if sentence:
    # Get sentiment prediction
    y_pred = clf.predict([sentence])[0]

    # Get probability
    y_proba = np.round(clf.predict_proba([sentence]), 2)

    # Text color based on prediction
    if y_pred == "negatief":
        color = "red"
    else:
        color = "green"

    # Display prediction
    st.markdown(f'Je text is <strong><span style="color: {color}">{y_pred}</span></strong> met een kans van {y_proba.max():.2f}', unsafe_allow_html=True)

# Hide streamlit menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
