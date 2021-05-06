import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import gensim

en_stop = set(nltk.corpus.stopwords.words('english'))


def tokenize(text):
    lda_tokens = []
    lda_tokens = gensim.utils.simple_preprocess(str(text), deacc=True)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [get_lemma(token) for token in tokens]
    tokens = [token for token in tokens if token not in en_stop]
    return tokens


#print('done')

