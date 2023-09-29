import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

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


def preprocess_text(text, remove_stopwords=False, stem=False, lemmatize=False):
    """Preprocess text by removing stopwords, stemming, and/or lemmatizing."""

    tokens = word_tokenize(text)

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


def feature_extraction(X_train, X_test, ngram_range=(1, 1)):
    """Extract features from text using count vectorizer."""

    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(X_train)
    X_train_feat = vectorizer.transform(X_train)
    X_test_feat = vectorizer.transform(X_test)
    
    return X_train_feat, X_test_feat


def classification(X_train, y_train, classifier, cv=5):
    """Classify data using different classifier and return the best hyperparameters."""

    # Define classifier and parameter grid for each case
    if classifier == 'svm':
        clf = SVC(kernel='linear')
        param_grid = {
            'C': [0.1, 1, 10, 100], # cost: smaller c --> wider margin
        }
    elif classifier == 'logistic_regression':
        clf = LogisticRegression(solver='saga', max_iter=5000)
        param_grid = {
            'C': [0.1, 0.5, 1, 5], #smoothing
        }
    elif classifier == 'naive_bayes':
        clf = MultinomialNB(force_alpha=True)
        param_grid = {
            'alpha': [0.1, 0.5, 1, 2]
        }
    
    # apply grid search with cross-validation
    grid_search = GridSearchCV(clf, param_grid, cv=cv) 
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    # get score on the whole training set
    train_score = best_clf.score(X_train, y_train)

    return best_clf, train_score, grid_search.best_params_


def plot_table(results, title):
    """Plot the results of the preprocessing options."""

    df = pd.DataFrame(results)

    # Create a table plot
    fig, ax = plt.subplots(figsize=(10, 4))   
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', cellColours=[['lightgray']*df.shape[1]] * df.shape[0])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    plt.title(title, fontsize=14)
    
    # Export the plot as an image
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    ## define parameters ##

    # data
    facts_file = 'data/facts.txt'
    fakes_file = 'data/fakes.txt'
    test_size = 0.2

    # feature extraction 
    ngram_range = (1, 1)

    # classification
    cv = 5

    # preprocessing options to compare
    preprocessing_options = list(product([True, False], repeat=3))
    results = []
    best_preprocessing = {
    'svm_best': {'cv_train_score': -1, 'test_score': -1, 'preprocessing_options': None, 'model_params': None},
    'svm_worst': {'cv_train_score': 2, 'test_score': 2, 'preprocessing_options': None, 'model_params': None},
    'logistic_regression_best': {'cv_train_score': -1, 'test_score': -1, 'preprocessing_options': None, 'model_params': None},
    'logistic_regression_worst': {'cv_train_score': 2, 'test_score': 2, 'preprocessing_options': None, 'model_params': None},
    'naive_bayes_best': {'cv_train_score': -1, 'test_score': -1, 'preprocessing_options': None, 'model_params': None},
    'naive_bayes_worst': {'cv_train_score': 2, 'test_score': 2, 'preprocessing_options': None, 'model_params': None},
    }

    all_facts, labels = load_data(facts_file, fakes_file)

    # run experiments for different preprocessing options
    for options in preprocessing_options:

        # prep data
        remove_stopwords, stem, lemmatize = options
        preprocessed_facts = [preprocess_text(fact, remove_stopwords, stem, lemmatize) for fact in all_facts]
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_facts, labels, test_size=test_size, random_state=42)
        X_train, X_test = feature_extraction(X_train, X_test, ngram_range)
        
        # classification
        clf_svm, train_score_svm, svm_best_param = classification(X_train, y_train, 'svm', cv)
        clf_lr, train_score_lr, lr_best_param = classification(X_train, y_train, 'logistic_regression', cv)
        clf_nb, train_score_nb, nb_best_param = classification(X_train, y_train, 'naive_bayes', cv)

        # calculate scores
        svm_train_score = round(train_score_svm, 3) 
        svm_cv_train_score = round(np.mean(cross_val_score(clf_svm, X_train, y_train, cv=cv)), 3)
        svm_test_score = round(clf_svm.score(X_test, y_test), 3) 

        lr_train_score = round(train_score_lr, 3)  
        lr_cv_train_score = round(np.mean(cross_val_score(clf_lr, X_train, y_train, cv=cv)), 3)
        lr_test_score = round(clf_lr.score(X_test, y_test), 3) 

        nb_train_score = round(train_score_nb, 3) 
        nb_cv_train_score = round(np.mean(cross_val_score(clf_nb, X_train, y_train, cv=cv)), 3)
        nb_test_score = round(clf_nb.score(X_test, y_test), 3) 

        # save all results
        results.append({
            'Stop rmv, Stem, Lemma': ', '.join(str(item) for item in options),
            'Train / CV Mean Train / Test': f"{nb_train_score} / {nb_cv_train_score} / {nb_test_score}",
            'Train / CV Mean Train / Test': f"{lr_train_score} / {lr_cv_train_score} / {lr_test_score}",
            'Train / CV Mean Train / Test': f"{svm_train_score} / {svm_cv_train_score} / {svm_test_score}"
        })

        # save best and worst results
        if svm_cv_train_score > best_preprocessing['svm_best']['cv_train_score']:
            best_preprocessing['svm_best']['model_params'] = svm_best_param
            best_preprocessing['svm_best']['preprocessing_options'] = options
            best_preprocessing['svm_best']['cv_train_score'] = svm_cv_train_score
            best_preprocessing['svm_best']['test_score'] = svm_test_score
        elif svm_cv_train_score < best_preprocessing['svm_worst']['cv_train_score']:
            best_preprocessing['svm_worst']['model_params'] = svm_best_param
            best_preprocessing['svm_worst']['preprocessing_options'] = options
            best_preprocessing['svm_worst']['cv_train_score'] = svm_cv_train_score
            best_preprocessing['svm_worst']['test_score'] = svm_test_score

        if lr_cv_train_score > best_preprocessing['logistic_regression_best']['cv_train_score']:
            best_preprocessing['logistic_regression_best']['model_params'] = lr_best_param
            best_preprocessing['logistic_regression_best']['preprocessing_options'] = options
            best_preprocessing['logistic_regression_best']['cv_train_score'] = lr_cv_train_score
            best_preprocessing['logistic_regression_best']['test_score'] = lr_test_score
        elif lr_cv_train_score < best_preprocessing['logistic_regression_worst']['cv_train_score']:
            best_preprocessing['logistic_regression_worst']['model_params'] = lr_best_param
            best_preprocessing['logistic_regression_worst']['preprocessing_options'] = options
            best_preprocessing['logistic_regression_worst']['cv_train_score'] = lr_cv_train_score
            best_preprocessing['logistic_regression_worst']['test_score'] = lr_test_score

        if nb_cv_train_score > best_preprocessing['naive_bayes_best']['cv_train_score']:
            best_preprocessing['naive_bayes_best']['model_params'] = nb_best_param
            best_preprocessing['naive_bayes_best']['preprocessing_options'] = options
            best_preprocessing['naive_bayes_best']['cv_train_score'] = nb_cv_train_score
            best_preprocessing['naive_bayes_best']['test_score'] = nb_test_score
        elif nb_cv_train_score < best_preprocessing['naive_bayes_worst']['cv_train_score']:
            best_preprocessing['naive_bayes_worst']['model_params'] = nb_best_param
            best_preprocessing['naive_bayes_worst']['preprocessing_options'] = options
            best_preprocessing['naive_bayes_worst']['cv_train_score'] = nb_cv_train_score
            best_preprocessing['naive_bayes_worst']['test_score'] = nb_test_score

        print(f"Stop rmv: {remove_stopwords}, Stem: {stem}, Lemma: {lemmatize}")
        print(f"Best parameters for SVM: {svm_best_param}")
        print(f"Best parameters for Logistic Regression: {lr_best_param}")
        print(f"Best parameters for Naive Bayes: {nb_best_param}")
        print("\n")

    # plot a table for the best and worst preprocessing options
    df_concise = pd.DataFrame(best_preprocessing)
    plot_table(df_concise, title='Best and Worst Preprocessing Options')

    # plot results from all preprocessing options
    df = pd.DataFrame(results)
    plot_table(results, title='Model Performance with Different Preprocessing Options')