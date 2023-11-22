from data_processing import preprocess
from evaluation import accuracy
from loader import load_instances, load_key
from itertools import product
from models import most_frequent_sense, nltk_lesk, bert_prediction
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
from data_creation import create_brown_data, create_seed_set
from models import nb_bootstrap, bootstrapping_grid_search

from itertools import product



if __name__ == "__main__":
    ### Data Loading ###
    data_f = '../data/multilingual-all-words.en.xml'
    key_f = '../data/wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for k, v in dev_instances.items() if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}

    ### Preprocessing Grid Search for Lesk ###
    print("Preprocessing Grid Search (Lesk)")

    lemmatize_options = [True, False]
    rmv_stop_words_options = [True, False]
    lowercase_options = [True, False]
    rmv_punctuation_options = [True, False]

    best_lemmatize = None
    best_rmv_stop_words = None
    best_lowercase = None
    best_rmv_punctuation = None
    best_accuracy = 0
    lesk_predictions = {}
    
    for lemmatize, rmv_stop_words, lowercase, rmv_punctuation in product(lemmatize_options, rmv_stop_words_options, lowercase_options, rmv_punctuation_options):
        # reset predictions
        lesk_predictions = {}

        # apply to dev set
        for instance_id, instance in dev_instances.items():
            context_sentence = preprocess(' '.join(instance.context), lemmatize=lemmatize, rmv_stop_words=rmv_stop_words, lowercase=lowercase, rmv_punctuation=rmv_punctuation)
            
            lesk_sense = nltk_lesk(context_sentence, instance.lemma)
            lesk_predictions[instance_id] = lesk_sense

        current_accuracy = accuracy(lesk_predictions, dev_key)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_lemmatize = lemmatize
            best_rmv_stop_words = rmv_stop_words
            best_lowercase = lowercase
            best_rmv_punctuation = rmv_punctuation

    # print best preprocessing options
    print(f"Best Parameters (Lesk): Lemmatize={best_lemmatize}, Remove Stop Words={best_rmv_stop_words}, Lowercase={best_lowercase}, Remove Punctuation={best_rmv_punctuation}")
    print("\n")

    ### Grid Search for Multinomial NB with Bootstrapping ###
    print("Grid Search for Multinomial NB with Bootstrapping")

    # Define the range of values for each parameter 
    # param_grid = {
    #     'n_top_lemmas': [10, 20, 50, 100], # max is 765 (number of distinct lemmas in dev and test set combined)
    #     'n_senses': [1, 2, 3],
    #     'n_sentences': [10000],
    #     'window_size': [3, 5, 10],
    #     'num_iterations': [3, 5],
    #     'confidence_threshold': [0.7, 0.75, 0.8],
    #     'fit_prior': [False, True],
    #     'alpha': [0.5, 1.0, 1.5]
    # }

    param_grid = {
        'n_top_lemmas': [50], # max is 765 (number of distinct lemmas in dev and test set combined)
        'n_senses': [1],
        'n_sentences': [100000],
        'window_size': [3],
        'num_iterations': [3],
        'confidence_threshold': [0.7],
        'fit_prior': [False],
        'alpha': [1.0]
    }

    # print number of combinations
    print(f"Number of combinations: {np.prod([len(values) for values in param_grid.values()])}")

    best_score = 0
    best_params = {}
    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        score = bootstrapping_grid_search(dev_key, dev_instances, test_instances, **params_dict)
        if score > best_score:
            best_score = score
            best_params = params_dict

    print(f"Best Parameters (Multinomial NB with Bootstrapping): {best_params}")
    print(f"Best Score (Multinomial NB with Bootstrapping): {best_score}")
    print("\n")

    ### Train best Multinomial NB with Bootstrapping ###
    print("Train best Multinomial NB with Bootstrapping")

    # get the best parameters from the grid search
    n_top_lemmas = best_params['n_top_lemmas']
    n_senses = best_params['n_senses']
    n_sentences = best_params['n_sentences']
    window_size = best_params['window_size']
    num_iterations = best_params['num_iterations']
    confidence_threshold = best_params['confidence_threshold']
    fit_prior = best_params['fit_prior']
    alpha = best_params['alpha']

    # prepare the seed set
    seed_data = create_seed_set(dev_instances, test_instances, n_top_lemmas, n_senses)
    for instance in seed_data: 
        preprocessed_context = preprocess(instance.context , lemmatize=True, rmv_stop_words=True, lowercase=True, rmv_punctuation=True)
        instance.context = " ".join(preprocessed_context)

    df_seed = pd.DataFrame([{'lemma': instance.lemma, 'context': instance.context, 'synset_name': instance.label} for instance in seed_data])
    df_seed.to_csv('../data/seed_data.csv', index=False)
    print(f"Seed Set: {len(seed_data)} instances, {df_seed.groupby(['lemma', 'synset_name']).size().mean()} instances per lemma-sense combination")

    # prepare the external dataset
    unique_lemmas = df_seed['lemma'].unique().tolist()
    unlabelled_data = create_brown_data(n_sentences=n_sentences, unique_lemmas=unique_lemmas, window_size=window_size)
    for instance in unlabelled_data:
        preprocessed_context = preprocess(instance.context , lemmatize=True, rmv_stop_words=True, lowercase=True, rmv_punctuation=True)
        instance.context = " ".join(preprocessed_context)

    df_unlabelled = pd.DataFrame([{'lemma': instance.lemma, 'context': instance.context} for instance in unlabelled_data])
    df_unlabelled.to_csv('../data/unlabelled_data.csv', index=False)

    print(f"Unlabelled Data: {len(unlabelled_data)} instances, {df_unlabelled.groupby(['lemma']).size().mean()} instances per lemma")

    # bootstrap the classifier
    nb_classifier, vectorizer = nb_bootstrap(seed_data, unlabelled_data, num_iterations=num_iterations, confidence_threshold=confidence_threshold, fit_prior=fit_prior, alpha=alpha, verbose=True)
    max_accuracy = len([instance for instance in test_instances.values() if instance.lemma in unique_lemmas]) / len(test_instances)
    print("\n")

    ### Prediction on Test Set for all methods ###
    print("Prediction on Test Set for all methods")
    baseline_predictions = {}
    lesk_predictions = {}
    bert_predictions = {}
    nb_predictions = {}

    for instance_id, instance in test_instances.items():
        context_sentence = preprocess(' '.join(instance.context), lemmatize=best_lemmatize, rmv_stop_words=best_rmv_stop_words, lowercase=best_lowercase, rmv_punctuation=best_rmv_punctuation)
        
        baseline_sense = most_frequent_sense(instance.lemma)
        lesk_sense = nltk_lesk(context_sentence, instance.lemma)
        bert_sense = bert_prediction(' '.join(instance.context), instance.lemma)
        nb_sense = nb_classifier.predict(vectorizer.transform([f"{instance.lemma} {' '.join(context_sentence)}"]))[0]


        baseline_predictions[instance_id] = baseline_sense
        lesk_predictions[instance_id] = lesk_sense
        bert_predictions[instance_id] = bert_sense
        nb_predictions[instance_id] = wn.synset(nb_sense)


    # save predictions to file  
    with open('../out/baseline_predictions.txt', 'w') as f:
        for instance_id, sense_key in baseline_predictions.items():
            f.write(f"{instance_id} {sense_key}\n")
    with open('../out/lesk_predictions.txt', 'w') as f:
        for instance_id, sense_key in lesk_predictions.items():
            f.write(f"{instance_id} {sense_key}\n")
    with open('../out/bert_predictions.txt', 'w') as f:
       for instance_id, sense_key in bert_predictions.items():
           f.write(f"{instance_id} {sense_key}\n")
    with open('../out/nb_predictions.txt', 'w') as f:
        for instance_id, sense_key in nb_predictions.items():
            f.write(f"{instance_id} {sense_key}\n")


    ### Evaluation ###
    baseline_accuracy = accuracy(baseline_predictions, test_key)
    lesk_accuracy = accuracy(lesk_predictions, test_key)
    bert_accuracy = accuracy(bert_predictions, test_key)
    nb_accuracy = accuracy(nb_predictions, test_key)

    print(f"Baseline Method: {baseline_accuracy}")
    print(f"Lesk Method: {lesk_accuracy}")
    print(f"BERT Method: {bert_accuracy}")
    print(f"Naive Bayes Method: {nb_accuracy} (Max Accuracy: {max_accuracy})")
