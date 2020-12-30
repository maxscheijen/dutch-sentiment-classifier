import pandas as pd
import numpy as np


def fi_trained_model(model):
    feat = model.named_steps['text_processors'][0].get_feature_names()
    coef = model.named_steps['classifier'].coef_.flatten()
    fi = pd.DataFrame([*zip(feat, coef)], columns=["Woord", "Belangrijkheid"])
    return fi


def feature_importance_prediction(model, text, cuttoff=0.5):
    fi = fi_trained_model(model)
    dense_features = (model.named_steps['text_processors']
                      .transform([text])
                      .toarray()
                      .flatten())
    bool_mask = np.ma.make_mask(dense_features)
    fi_predict = (fi.iloc[bool_mask].sort_values("Belangrijkheid",
                                                 ascending=False))
    fi_predict['Sentiment'] = np.where(fi_predict['Belangrijkheid'] > 0,
                                       "Positief", "Negatief")
    fi_predict = fi_predict.loc[(fi_predict["Belangrijkheid"] > cuttoff) |
                                (fi_predict["Belangrijkheid"] < -cuttoff)]
    fi_predict['Belangrijkheid'] = fi_predict['Belangrijkheid'].round(2)

    return fi_predict.reset_index(drop=True)


def feature_importance_in_text(fi_df, text):

    replace_dict = pd.Series(
        np.where(fi_df.Sentiment.str.contains("Positief"), '<b><font color="green">' +
                 fi_df.Woord + '</font></b>', '<b><font color="red">' + fi_df.Woord + '</font></b>'),
        index=fi_df.Woord.values).to_dict()

    for word, replace in replace_dict.items():
        text = text.replace(" " + word.lower() + " ",
                            " " + replace.lower() + " ")

    return text
