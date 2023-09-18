import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from itertools import product

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def load_data(facts_file, fakes_file):
    """Load data from two files and return a list of facts and a list of fakes."""
    with open(facts_file, 'r', encoding='utf-8') as f:
        facts = f.read().splitlines()

    with open(fakes_file, 'r', encoding='utf-8') as f:
        fakes = f.read().splitlines()

    all_facts = facts + fakes

    # create labels (1 for facts, 0 for fakes)
    labels = np.concatenate((np.ones(len(facts)), np.zeros(len(fakes))), axis=0)

    return all_facts, labels


def preprocess_text(text, tokenize=False, remove_stopwords=False, stem=False, lemmatize=False):
    """Preprocess text by tokenizing, removing stopwords, stemming, and/or lemmatizing."""

    if tokenize:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def feature_extraction(X_train, X_test, feature_extractor='tfidf', max_features=1000, ngram_range=(1, 1)):
    """Extract features from text using TF-IDF vectorizer."""

    if feature_extractor == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        vectorizer.fit(X_train)
        X_train_feat = vectorizer.transform(X_train)
        X_test_feat = vectorizer.transform(X_test)

    elif feature_extractor == 'count':
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        vectorizer.fit(X_train)
        X_train_feat = vectorizer.transform(X_train)
        X_test_feat = vectorizer.transform(X_test)

    else:
        raise ValueError('Feature extractor not supported.')
    
    return X_train_feat, X_test_feat


def classification(X_train, y_train, classifier, cv=5, kernel='linear'):
    """Classify data using different classifier."""

    if classifier == 'svm':
        clf = SVC(kernel=kernel)
    elif classifier == 'logistic_regression':
        clf = LogisticRegression()
    elif classifier == 'naive_bayes':
        clf = MultinomialNB()
    else:
        raise ValueError('Classifier not supported.')
    
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv)
    clf.fit(X_train, y_train)

    return clf, cv_scores

def plot_preprocessing_options_scores(results):
    """Plot the results of the preprocessing options."""

    # plot the results
    df = pd.DataFrame(results)

    # Create a table-like plot
    fig, ax = plt.subplots(figsize=(10, 4))   
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', cellColours=[['lightgray']*df.shape[1]] * df.shape[0])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # You can adjust the scaling as needed
    
    # Export the plot as an image (e.g., PNG)
    plt.savefig('preprocessing_scores.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    # define parameters
    facts_file = 'data/facts.txt'
    fakes_file = 'data/fakes.txt'
    test_size = 0.15

    # feature extraction 
    feature_extractor = 'count'
    max_features = 1000
    ngram_range = (1, 1)

    # classification
    cv = 5
    kernel = 'linear' # svm only

    # Define the parameter combinations to test
    preprocessing_options = list(product([True, False], repeat=4))

    # Initialize a list to store the results
    results = []

    for options in preprocessing_options:

        tokenize, remove_stopwords, stem, lemmatize = options
        
        # load data
        all_facts, labels = load_data(facts_file, fakes_file)
        preprocessed_facts = [preprocess_text(fact, tokenize, remove_stopwords, stem, lemmatize) for fact in all_facts]
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_facts, labels, test_size=test_size, random_state=42)
        X_train, X_test = feature_extraction(X_train, X_test, feature_extractor, max_features, ngram_range)
        
        # classification
        clf_svm, cv_scores_svm = classification(X_train, y_train, 'svm', cv, kernel)
        clf_lr, cv_scores_lr = classification(X_train, y_train, 'logistic_regression', cv)
        clf_nb, cv_scores_nb = classification(X_train, y_train, 'naive_bayes', cv)

        # calculate the mean cross-validation scores
        svm_score = round(clf_svm.score(X_test, y_test), 3)
        lr_score = round(clf_lr.score(X_test, y_test), 3)
        nb_score = round(clf_nb.score(X_test, y_test), 3)

        results.append({
            'Tokenize, Stop rmv, Stem, Lemma': ', '.join(str(item) for item in options),
            'SVM Score': svm_score,
            'LR Score': lr_score,
            'NB Score': nb_score
        })

    # plot the results
    df = pd.DataFrame(results)
    plot_preprocessing_options_scores(results)

    # TODO: Evaluate feature extraction options (also n-grams)
    # TODO: Check cross validation
    # TODO: Print/plot more results