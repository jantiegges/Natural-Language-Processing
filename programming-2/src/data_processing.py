from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(context, lemmatize=False, rmv_stop_words=False, lowercase=False, rmv_punctuation=False):
    """
    Tokenize, and optionally lemmatize, remove stop words, convert to lowercase, and remove punctuation from a context.
    """
    tokens = word_tokenize(context)

    if rmv_punctuation:
        tokens = [token for token in tokens if token.isalpha()]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if rmv_stop_words:
        tokens = [token for token in tokens if token not in stop_words]

    if lowercase:
        tokens = [token.lower() for token in tokens]

    return tokens