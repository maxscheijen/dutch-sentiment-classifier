import pandas as pd
import numpy as np


def fi_trained_model(model):
    feat = model.named_steps['text_processors'][0].get_feature_names()
    coef = model.named_steps['classifier'].coef_.flatten()
    fi = pd.DataFrame([*zip(feat, coef)], columns=["Woord", "Belangrijkheid"])
    return fi


def feature_importance_prediction(model, text):
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
    return fi_predict.reset_index(drop=True)
