import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

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

    # Stratified cross-validator
    kfold = StratifiedKFold(n_splits=config.CV_SPLITS,
                            random_state=config.SEED,
                            shuffle=True)

    # Overall metrics to track
    cv_metrics = {
        "acc": [],
        "auc": [],
        "f1": []
    }

    # Cross validation loop
    for train_index, valid_index in tqdm(kfold.split(X=X, y=y)):
        # Create train and validation dataset
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # Train model
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_valid)

        # Calculate metrics
        cv_metrics["acc"].append(metrics.accuracy_score(y_valid, y_pred))
        cv_metrics["auc"].append(metrics.roc_auc_score(y_valid, y_pred))
        cv_metrics["f1"].append(metrics.f1_score(y_valid, y_pred))

    # Save metrics as json
    metrics_df = pd.DataFrame(cv_metrics).mean()
    metrics_df.to_json(config.METRIC_FILE)
    print(metrics_df)


if __name__ == "__main__":
    run_cross_validation()
