import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, brown
from data_processing import preprocess
from collections import Counter

lemmatizer = WordNetLemmatizer()

class TrainingInstance:
    def __init__(self, lemma, context, label):
        self.id = None
        self.lemma = lemma
        self.context = context
        self.label = label

def create_seed_set(dev_instances, test_instances, n_top_lemmas=None, n_senses=None):
    """Create a seed set with the top n lemmas from the dev and test sets."""
    # combine the dev and test instances
    instances = dev_instances.copy()
    instances.update(test_instances)

    # get the top n lemmas. if n is None, get all lemmas
    if n_top_lemmas is None:
        n_top_lemmas = len(instances)
    top_lemmas = __extract_lexical_items(instances, n_top_lemmas)
    
    # create seed set
    seed_set = []
    for lemma in top_lemmas:
        seed_set.extend(__create_seed_data_for_lemma(lemma, n_senses))

    return seed_set

def create_brown_data(n_sentences=1000, unique_lemmas=None, window_size=10):
    """Create a list of TrainingInstance objects from the Brown corpus but only for the given lemmas."""
    unlabelled_data = []
    for sentence in brown.sents()[:n_sentences]:
        for lemma, context in __extract_noun_lemma_and_context(" ".join(sentence), window_size=window_size):
            if unique_lemmas is None or lemma in unique_lemmas:
                context = preprocess(context, lemmatize=True, rmv_stop_words=True, lowercase=True, rmv_punctuation=True)
                context = " ".join(context)
                unlabelled_data.append(TrainingInstance(lemma, context, None))

    return unlabelled_data

def __get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}

    return tag_dict.get(tag, wn.NOUN)

def __extract_noun_lemma_and_context(sentence, window_size=10):
    """Extract the lemma and context for each noun in a sentence."""
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    for i, (word, pos) in enumerate(pos_tags):
        # check if the word is a noun
        if pos.startswith('N'):
            # lemmatize the word based on its POS tag
            lemma = lemmatizer.lemmatize(word, __get_wordnet_pos(word))
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(tokens))
            context = " ".join(tokens[start:end])
            # yield the lemma and context
            yield lemma, context


def __extract_lexical_items(instances, n_top_lemmas):
    """Extract the top n lemmas from the instances."""
    lemmas = [instance.lemma for instance in instances.values()]
    lemma_counts = Counter(lemmas)
    top_lemmas = [lemma for lemma, count in lemma_counts.most_common(n_top_lemmas)]
    return top_lemmas


def __create_seed_data_for_lemma(lemma, n_senses=None):
    """Create labelled data for a given lemma using WordNet."""
    seed_data = []
    added_contexts = set() 

    synsets = wn.synsets(lemma)
    if n_senses is None:
        n_senses = len(synsets)

    for synset in synsets[:n_senses]:
        # create different contexts for each sense from wordnet examples
        contexts = {
            'definition': [synset.definition()],
            'examples': synset.examples(),
            'hypernyms': [example for hypernym in synset.hypernyms() for example in hypernym.examples()],
            'hyponyms': [example for hyponym in synset.hyponyms() for example in hyponym.examples()],
            'synonyms': [example for lemma_syn in synset.lemmas() for example in lemma_syn.synset().examples()]
        }

        for context_list in contexts.values():
            for context in context_list:
                context_key = (synset.name(), context) # only add unique lemma-context combinations
                
                if context_key not in added_contexts:
                    instance = TrainingInstance(lemma, context, synset.name())
                    seed_data.append(instance)
                    added_contexts.add(context_key)

    return seed_data
