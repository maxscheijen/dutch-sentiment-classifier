import re
import pandas as pd
import numpy as np


def fi_trained_model(model) -> pd.DataFrame:
    """"Get feature importance from trained model"""
    # Get feature names
    feat = model.named_steps['text_processors'][0].get_feature_names()

    # Get model (model) coefficients store them in DataFrame
    coef = model.named_steps['classifier'].coef_.flatten()
    fi = pd.DataFrame([*zip(feat, coef)], columns=["Woord", "Belangrijkheid"])
    return fi


def feature_importance_prediction(model, text: str, cuttoff: float = 0.5):
    """Match feature importance of trained model to model predictions"""
    # Get feature importance from trained model
    fi = fi_trained_model(model)

    # Get dense features from sparse matrix
    dense_features = (model.named_steps['text_processors']
                      .transform([text])
                      .toarray()
                      .flatten())

    # Create boolean mask to select features
    bool_mask = np.ma.make_mask(dense_features)

    # Sort words on feature importance
    fi_predict = (fi.iloc[bool_mask].sort_values("Belangrijkheid",
                                                 ascending=False))

    # Calculate sentiment label
    fi_predict['Sentiment'] = np.where(fi_predict['Belangrijkheid'] > 0,
                                       "Positief", "Negatief")

    # Select dataframe based on feature importance threshold
    fi_predict = fi_predict.loc[(fi_predict["Belangrijkheid"] > cuttoff) |
                                (fi_predict["Belangrijkheid"] < -cuttoff)]

    # Round feature importance
    fi_predict['Belangrijkheid'] = fi_predict['Belangrijkheid'].round(2)
    fi_predict = fi_predict.reset_index(drop=True)

    # Uppercase first character if is start of sentence
    for word in text.split(" "):

        # Check if first character of first is upper case
        if word[0].isupper() and word.lower() in fi["Woord"].values:

            # Replace lower case with title case
            fi_predict["Woord"] = fi_predict["Woord"].replace(
                word.lower(), word.title())

    return fi_predict


def feature_importance_in_text(fi_df: pd.DataFrame, text: str) -> str:
    """Combine original text with sentiment words"""
    replace_dict = pd.Series(
        np.where(fi_df.Sentiment.str.contains("Positief"), '<b><font color="green">' +
                 fi_df.Woord + '</font></b>', '<b><font color="red">' + fi_df["Woord"] + '</font></b>'),
        index=fi_df["Woord"].values).to_dict()

    # Replace original text word with sentiment meta data words
    for word, replace in replace_dict.items():
        text = re.sub(" " + r"[a-zA-Z]" + word[1:] + " ",
                            " " + replace + " ", text)

    return text
