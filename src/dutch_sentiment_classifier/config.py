import dutch_sentiment_classifier

from pathlib import Path


BASE_DIR = (Path(dutch_sentiment_classifier.__file__)
            .resolve().parent.parent.parent)

DATA_DIR = BASE_DIR / "data"
TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

SEED = 123

SPLIT_SIZE = 0.7