import nltk as nltk
import numpy as np
import pandas as pd
import collections as coll

from sklearn.base import TransformerMixin

# Hyperpameters for n-grams:
#   - n = 1:4
#   - multimap = [True, False]
#   - low-count cutoff
#   - 
###################################################################################
# Produce new representation of data such that each n-gram is associated with 
# the number of times it is observed in a text of a particular language.
#
# I can think of two ways to encode using n-grams.
#
# Method #1: For each language, calculate the # of texts in which each n-gram
#            has appeared.  This means that an n-gram is counted <once> per text.
#
# Method #2: For each language, calculate the # of occurrences of each n-gram across
#            all texts.  This means that an n-gram is counted <multiple times> per text.

class NGramGenerator(TransformerMixin):
    def __init__(self, n=1, multimap=True, count_threshold=0, verbose=False):
        self.n = n
        self.multimap = multimap
        self.count_threshold = count_threshold,
        self.verbose = verbose

    # Extract n-grams from text snippit
    #
    # Input:
    #   - String s i.e. "okok"
    #   - Integer n i.e. 2
    # Output:
    #   - List<String> i.e. ["ok","ok"]

    def string_to_ngrams(self, s):
        text = str(s).decode('utf-8').lower()
        text = text.replace(' ', '') # remove spaces
        ngrams = nltk.ngrams([c for c in text], self.n)
        return [''.join(g) for g in ngrams]

    def transform(self, X):
        Z = {}
        # Construct hash of arrays.
        for index, row in X.iterrows():
            # Code the language of the observation
            category = np.zeros(3)
            category[row['Category']] = 1
            # Break the text into n-grams
            ngrams = self.string_to_ngrams(row['Text'])
            if not self.multimap:
                ngrams = list(set(ngrams))
            for ngram in ngrams:
                if ngram in Z:
                    # Sum element-wise with entries.
                    Z[ngram] = Z[ngram] + category # for some reason += works by reference and glitches
                else:
                    Z[ngram] = category
                if self.verbose:
                    print("%s:%s" % (ngram, Z[ngram]))
        # Convert into data frame   
        Z = pd.DataFrame(Z).transpose()
        # Filter low counts
        keep = Z.apply(lambda row: sum(row) >= self.count_threshold, axis = 1)
        Z = Z[keep == 1]

        return Z

    def fit(self, *_):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
