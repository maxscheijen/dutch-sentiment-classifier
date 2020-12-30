import joblib
import streamlit as st
import numpy as np
import seaborn as sns

from classifier import config
from classifier.utils.feature_importance import feature_importance_prediction, feature_importance_in_text

st.beta_set_page_config(page_title="Nederlandse Sentiment Classificatie",
                        page_icon="https://raw.githubusercontent.com/maxscheijen/maxscheijen.github.io/main/favicon.ico")

# Load model
# @st.cache(allow_output_mutation=True)


def load_model(path):
    clf = joblib.load(path)
    return clf


clf = load_model(config.MODEL_NAME)

# Header
st.markdown("<h1>Sentiment Classificatie</h1>", unsafe_allow_html=True)

# Get user input
sentence = str(st.text_area(label='Voer hier uw text in:'))

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
    st.markdown(
        f'Je text is <strong><span style="color: {color}">{y_pred}</span></strong> met een kans van {y_proba.max():.0%}.', unsafe_allow_html=True)

# Detail header
st.write("<h2>Meer informatie</h2>", unsafe_allow_html=True)

# Detal description
st.write("Door de knop hieronder aan te vinken krijg je een tabel te zien. Deze tabel toont welke woorden invloed hebben op het sentiment van de text. Daarnaast laat deze tabel ook zie in hoe sterk het woord bijdraagt in de bepaling van het sentiment ze zien in de column met de naam \"Belangrijkheid\".")

# Display detailed sentiment information
more_info = st.checkbox("Laat zien!")

if more_info:
    if len(sentence) == 0:
        st.write("Je moet eerst een text invoeren!")
    else:
        st.markdown("<h3>Gemarkeerde worden op bassis van sentiment</h3>",
                    unsafe_allow_html=True)
        threshold = st.slider(min_value=0., max_value=6., value=3., step=0.01,
                              label="Belangrijkheid drempel waarde")

        # Calculate feature importance
        fi_sentence = feature_importance_prediction(
            clf, sentence, cuttoff=threshold)

        # Display text with sentiment
        sentiment_text = feature_importance_in_text(fi_sentence, sentence)
        st.markdown(sentiment_text, unsafe_allow_html=True)

        st.write("<h3>Woorden en hun invloed op sentiment</h3>",
                 unsafe_allow_html=True)

        # Create color map
        cmap = sns.diverging_palette(h_neg=10, h_pos=147, s=74, l=50, sep=10,
                                     n=25, as_cmap=True)
        min_color = fi_sentence.min()["Belangrijkheid"]
        max_color = fi_sentence.max()["Belangrijkheid"]
        v_color_value = np.array([abs(min_color), max_color]).max()
        fi_sentence_color = (fi_sentence.style
                             .background_gradient(cmap,
                                                  vmin=-v_color_value,
                                                  vmax=v_color_value,
                                                  axis=1))

        # Display dataframe
        st.dataframe(fi_sentence_color, height=((100//3)+100)*len(fi_sentence))

# Hide streamlit menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
