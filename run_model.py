import pandas as pd
import numpy as np
import nltk
import keras
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K
from argparse import ArgumentParser
from sklearn.externals import joblib


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', help='Path to the input CSV')
    parser.add_argument('--submission', help='Path to the submission CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    vector = joblib.load('vector.pkl')
    pipeline = joblib.load('model.pkl')
    X = df['text']
    features = vector.transform(X.astype('U'))
    predictions = pipeline.predict(features)
    predictions = df.join(pd.DataFrame(predictions))
    predictions.drop('text', inplace=True, axis=1)
    predictions.columns = ['review_id', 'stars_1', 'stars_2',
                           'stars_3', 'stars_4', 'stars_5']
    predictions.to_csv(args.submission, index=False)

if __name__ == '__main__':
    main()
