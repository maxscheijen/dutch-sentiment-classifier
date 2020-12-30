import pandas as pd

from sklearn.model_selection import cross_validate

from classifier import config
from classifier.sentiment_model import SentimentClassifier


def run_cross_validation() -> None:
    # Load data
    train = pd.read_csv(config.TRAIN_DATA)

    # Encode labels for evaluation
    label_mapping = {
        "negatief": 0,
        "positief": 1,
    }

    # Features and target
    X = train.text.values
    y = train.sentiment.map(label_mapping).values

    # Declare model
    clf = SentimentClassifier()

    # Cross validation
    cv = cross_validate(estimator=clf,
                        X=X, y=y, cv=config.CV_SPLITS,
                        scoring=config.METRICS,
                        verbose=2, n_jobs=-1)

    # Mean model metrics
    metrics = pd.DataFrame(cv).mean().round(4)
    metrics.to_json(config.METRIC_FILE)
    print(metrics)


if __name__ == "__main__":
    run_cross_validation()
