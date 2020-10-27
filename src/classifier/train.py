import pandas as pd

from classifier import config
from classifier.sentiment_model import SentimentClassifier


def run_training():
    # Load data
    train = pd.read_csv(config.TRAIN_DATA)

    # Features and target
    X = train.text.values
    y = train.sentiment.values

    # Declare model
    clf = SentimentClassifier()

    # Train the model
    clf.fit(X, y)

    # Save the model
    clf.save(config.MODEL_NAME)


if __name__ == "__main__":
    run_training()
