import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from dutch_sentiment_classifier import config


# Load stopwords
stop_words = pd.read_csv(
    "https://raw.githubusercontent.com/xiamx/node-nltk-stopwords/master/data/stopwords/dutch"
    ).values.flatten().tolist()

# TF-IDF vectorizer
tfidf = TfidfVectorizer(strip_accents="unicode",
                        stop_words=stop_words,
                        ngram_range=config.NGRAM_RANGE,
                        max_df=config.MAX_DF,
                        min_df=config.MIN_DF,
                        max_features=config.VOCAB_SIZE,
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True)

# Text processing pipeline
text_processors = Pipeline(steps=[
    ("tfidf", tfidf)
])
