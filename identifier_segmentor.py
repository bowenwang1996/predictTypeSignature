import re
from nltk.stem.snowball import SnowballStemmer

SEGMENTOR = re.compile(r'(?:^[^A-Za-z0-9\s])?(?:[^a-z\_\s]+$|[^a-z\_\s]+[0-9\.]|[^a-z\_\s]+(?![a-z])|[A-Z][^A-Z0-9\_\s]+|[^A-Z0-9\_\s]+)')

#STEM_SIZE = 4
stemmer = SnowballStemmer("english")

def segment(ident):
    words = [stemmer.stem(w) for w in SEGMENTOR.findall(ident)]
    if len(words) == 0:
        return [stemmer.stem(ident)]
    return words
