from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from data_creation import TrainingInstance, create_seed_set, create_brown_data
from data_processing import preprocess
from itertools import product
import pandas as pd
from evaluation import accuracy


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def most_frequent_sense(word):
    """Get the most frequent sense of a word as per WordNet's ordering."""
    synsets = wn.synsets(word)
    return synsets[0] if synsets else None


def nltk_lesk(context_sentence, ambiguous_word):
    """Get word sense using NLTK's Lesk algorithm."""
    return lesk(context_sentence, ambiguous_word)


def bert_prediction(sentence, word):
    """
    Get the prediction for a word in a sentence using cosine similarity
    between the word's sense embeddings and the context embedding
    """
    context_embedding = __get_bert_embedding(sentence, word)
    best_sense = None
    max_similarity = float('-inf')

    for sense in wn.synsets(word):
        sense_embedding = __get_sense_embedding(sense)
        context_embedding_flat = context_embedding.detach().numpy().flatten()
        sense_embedding_flat = sense_embedding.detach().numpy().flatten()
        similarity = -cosine(context_embedding_flat, sense_embedding_flat)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_sense = sense

    return best_sense

def nb_bootstrap(seed_set, unlabelled_data, num_iterations=5, confidence_threshold=0.7, fit_prior=False, alpha=1.0, verbose=False):
    """Bootstrap a Naive Bayes classifier using a seed set and unlabelled data."""
    train_data = seed_set.copy()
    test_data = unlabelled_data.copy()

    for iteration in range(num_iterations):
        nb_classifier, vectorizer = __train_nb_classifier(train_data, fit_prior=fit_prior, alpha=alpha)

        # create combined string of lemma and context for each instance
        combined_features = [f"{instance.lemma} {instance.context}" for instance in test_data]
        test_data_vec = vectorizer.transform(combined_features)

        predictions = nb_classifier.predict(test_data_vec)
        probabilities = nb_classifier.predict_proba(test_data_vec)

        # get indices of instances with confidence above threshold
        confident_indices = [i for i, probs in enumerate(probabilities) if max(probs) > confidence_threshold]

        # add confidently predicted instances to train data
        for i in confident_indices:
            new_instance = TrainingInstance(test_data[i].lemma, test_data[i].context, predictions[i])
            train_data.append(new_instance)

        # remove confidently predicted instances from test data
        test_data = [instance for i, instance in enumerate(test_data) if i not in confident_indices]
    
        if verbose:
            print(f"Iteration {iteration}:")
            print(f"    Train data size: {len(train_data)}")
            print(f"    Unlabeled data size: {len(test_data)}")

    return nb_classifier, vectorizer

def bootstrapping_grid_search(dev_key, dev_instances, test_instances, n_top_lemmas, n_senses, n_sentences, window_size, num_iterations, confidence_threshold, fit_prior, alpha):
    """Grid search for the bootstrapping method."""
    # prepare the seed set
    seed_data = create_seed_set(dev_instances, test_instances, n_top_lemmas, n_senses)
    for instance in seed_data: 
        preprocessed_context = preprocess(instance.context , lemmatize=True, rmv_stop_words=True, lowercase=True, rmv_punctuation=True)
        instance.context = " ".join(preprocessed_context)
    df_seed = pd.DataFrame([{'lemma': instance.lemma, 'context': instance.context, 'synset_name': instance.label} for instance in seed_data])
    
    # prepare the external dataset
    unique_lemmas = df_seed['lemma'].unique().tolist()
    unlabelled_data = create_brown_data(n_sentences=n_sentences, unique_lemmas=unique_lemmas, window_size=window_size)
    for instance in unlabelled_data:
        preprocessed_context = preprocess(instance.context , lemmatize=True, rmv_stop_words=True, lowercase=True, rmv_punctuation=True)
        instance.context = " ".join(preprocessed_context)

    # bootstrap the classifier
    nb_classifier, vectorizer = nb_bootstrap(seed_data, unlabelled_data, num_iterations=num_iterations, confidence_threshold=confidence_threshold, fit_prior=fit_prior, alpha=alpha, verbose=False)

    # get predictions on dev set
    predictions = {}
    for instance_id, instance in dev_instances.items():
        context_sentence = preprocess(' '.join(instance.context), lemmatize=True, rmv_stop_words=True, lowercase=True)
        combined_features = f"{instance.lemma} {' '.join(context_sentence)}"
        test_data = vectorizer.transform([combined_features])
        prediction = nb_classifier.predict(test_data)[0]
        predictions[instance_id] = wn.synset(prediction)

    nb_accuracy = accuracy(predictions, dev_key)

    return nb_accuracy


def __get_bert_embedding(sentence, target_word):
    """Get the embedding for a target word in a sentence"""
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    target_word_index = tokenizer.convert_tokens_to_ids(target_word)
    target_embedding = embeddings[0, inputs['input_ids'][0] == target_word_index, :]
    return target_embedding.mean(dim=0)

def __get_sense_embedding(sense):
    """Get the embedding for a sense using its definition"""
    definition = sense.definition()
    example = sense.examples()[0] if sense.examples() else ""
    context = f"{definition} {example}"
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings.mean()

def __train_nb_classifier(train_data, fit_prior=False, alpha=1.0):
    """Train a Naive Bayes classifier """
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(instance.context for instance in train_data)
    y_train = [instance.label for instance in train_data]

    nb_classifier = MultinomialNB(fit_prior=fit_prior, alpha=alpha)
    nb_classifier.fit(X_train, y_train)

    return nb_classifier, vectorizer