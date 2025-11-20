# src/feature_engineering.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import load_params, logger
import os

def main():
    params = load_params()['feature_engineering']
    
    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")

    train_df['content'].fillna("", inplace=True)
    test_df['content'].fillna("", inplace=True)

    vectorizer_type = params['vectorizer']
    max_features = params['max_features']
    ngram_range = tuple(params['ngram_range'])

    if vectorizer_type == "count":
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    else:
        raise ValueError("vectorizer must be 'count' or 'tfidf'")

    X_train = vectorizer.fit_transform(train_df['content'])
    X_test = vectorizer.transform(test_df['content'])

    train_feat = pd.DataFrame(X_train.toarray())
    test_feat = pd.DataFrame(X_test.toarray())

    train_feat['label'] = train_df['sentiment'].values
    test_feat['label'] = test_df['sentiment'].values

    os.makedirs("data/features", exist_ok=True)
    train_feat.to_csv("data/features/train_bow.csv", index=False)
    test_feat.to_csv("data/features/test_bow.csv", index=False)

    logger.info(f"Feature engineering done with {vectorizer_type}, {max_features} features")

if __name__ == "__main__":
    main()