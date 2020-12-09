import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from classifier import config
from classifier.utils.process_data import text_processors


class SentimentClassifier:
    """Sentiment classifier that can be trained on text data with
    corresponding sentiment labels.
    """
    def __init___(self):
        self.clf = None

    def fit(self, X: pd.Series, y: pd.Series) -> "SentimentClassifier":
        """Fit the sentiment classification pipeline

        Parameters
        ----------
        X : pd.Series
            Pandas series containing text data
        y : Pandas series containg the sentiment label corresponding

        Returns
        -------
        SentimentClassifier
            Returns the trained sentiment classification model
        """
        # Declaring model pipeline
        self.clf = Pipeline(steps=[
            ("text_processors", text_processors),
            ("classifier", LogisticRegression())
        ])

        # Training sentiment model
        self.clf.fit(X, y)

        return self.clf

    def predict(self, X: pd.Series) -> np.array:
        """Predict sentiment based on input data

        Parameters
        ----------
        X : pd.Series
            Input data containg text

        Returns
        -------
        np.array
            Returns numpy array with sentiment prediction
        """
        # Get sentiment prediction
        y_pred = self.clf.predict(X)

        return y_pred

    def predict_proba(self, X) -> np.array:
        """Predict probabilities of sentiment input data

        Parameters
        ----------
        X : pd.Series
            Input data containg text

        Returns
        -------
        np.array
            Returns numpy array with sentiment prediction probabilities
        """
        # Get sentiment probabilities of sentiment classes
        y_proba = self.clf.predict_proba(X)

        return y_proba

    def save(self, path: str) -> None:
        """Save the trained sentiment model

        Parameters
        ----------
        path : str
            Save trained model path
        """
        # Create trained model directory
        config.TRAINED_MODEL_DIR.mkdir(exist_ok=True, parents=True)

        # Save model
        joblib.dump(self.clf, config.MODEL_NAME, compress=3)

    def load(self, path: str) -> "SentimentClassifier":
        """Load the trained sentiment model

        Returns
        -------
        [type]
            Returns the trained sentiment models
        """
        # Declare model instance
        model = SentimentClassifier()

        # Load model
        model.clf = joblib.load(path)

        return model.clf
