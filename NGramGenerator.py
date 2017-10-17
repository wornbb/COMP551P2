# # Load data
# 
# from sklearn.pipeline import Pipeline, FeatureUnion
# 
# X  = pd.read_csv("data/train_set_x.csv")
# Y  = pd.read_csv("data/train_set_y.csv")
# X  = pd.merge(X, Y, on = 'Id')
# Xn = pd.read_csv("data/test_set_x.csv")
# 
# # Split into n-grams
# 
# pipeline = Pipeline(
#     [( 'ng', NGramGenerator() )] +
# #   [( 'id', IdentityTransformer() )] +
#     []
# )
# A = pipeline.fit_transform(X[:1000])

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

class NGramGenerator(TransformerMixin):
    def __init__(self):
        None

        
        
    ###################################################################################
    # Extract n-grams from text snippit
    #
    # Input:
    #   - String s i.e. "okok"
    #   - Integer n i.e. 2
    # Output:
    #   - List<String> i.e. ["ok","ok"]

    def string_to_ngrams(self, s, n = 1):
        text = str(s).decode('utf-8').lower()
        text = text.replace(' ', '') # remove spaces
        ngrams = nltk.ngrams([c for c in text], n)
        return [''.join(g) for g in ngrams]

    
    
    ###################################################################################
    # Produce new representation of data such that each n-gram is associated with 
    # the number of times it is observed in a text of a particular language.
    #
    # Input:
    #   - pd.DataFrame train_set
    #   - Boolean multimap: True if an n-gram is counted multiple times per text, and False
    #                       if an n-gram is counted only once per text.
    #   - Integer count_threshold:
    #   - Boolean normalize
    # Output:
    #   - pd.DataFrame

    # I can think of two ways to encode using n-grams.
    #
    # Method #1: For each language, calculate the # of texts in which each n-gram
    #            has appeared.  This means that an n-gram is counted <once> per text.
    #
    # Method #2: For each language, calculate the # of occurrences of each n-gram across
    #            all texts.  This means that an n-gram is counted <multiple times> per text.

    def transform(self,
                  X,
                  Y = None,
                  multimap = True,
                  n = 1,
                  verbose = False,
                  count_threshold = 0,
                  normalize = False):
        
        Z = {}
        # Construct hash of arrays.
        for index, row in X.iterrows():
            # Code the language of the observation
            category = np.array([0, 0, 0, 0, 0])
            category[row['Category']] = 1
            # Break the text into n-grams
            ngrams = self.string_to_ngrams(row['Text'], n = 1)
            if not multimap:
                ngrams = list(set(ngrams))
            for ngram in ngrams:
                if ngram in Z:
                    # Sum element-wise with entries.
                    Z[ngram] = Z[ngram] + category # for some reason += works by reference and glitches
                else:
                    Z[ngram] = category
                if verbose:
                    print("%s:%s" % (ngram, Z[ngram]))
        # Convert into data frame   
        Z = pd.DataFrame(Z).transpose()
        # Filter low counts
        keep = Z.apply(lambda row: sum(row) >= count_threshold, axis = 1)
        Z = Z[keep == 1]
        # Normalize by sum of y for each col y in Y
        if normalize and Y != None:
            totals = coll.Counter(Y['Category'])
            for colname in totals:
                Z[colname] = Z[colname].apply(lambda x: 1. * x / totals[colname])
        # Return
        return Z
    
    def fit(self, *_):
        return self

