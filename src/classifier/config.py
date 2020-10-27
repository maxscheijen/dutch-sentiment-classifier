import classifier

from pathlib import Path

# ROOT
BASE_DIR = (Path(classifier.__file__)
            .resolve().parent.parent)

CLASSIFIER_DIR = BASE_DIR / "classifier"

# DATA
DATA_DIR = BASE_DIR.parent / "data"
TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

SEED = 123

SPLIT_SIZE = 0.7

# TEXT PROCESSING PARAMS
MAX_DF = 0.75
MIN_DF = 1
VOCAB_SIZE = 2000
NGRAM_RANGE = (1, 1)

# MODEL
TRAINED_MODEL_DIR = CLASSIFIER_DIR / "trained_model"
MODEL_NAME = TRAINED_MODEL_DIR / "sentiment_classifier.pkl"

# CROSS VALIDATION
CV_SPLITS = 5
METRICS_DIR = CLASSIFIER_DIR / "metrics"
METRIC_FILE = METRICS_DIR / "metrics.json"
