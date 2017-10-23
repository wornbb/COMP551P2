
from sklearn.feature_extraction.text import TfidfVectorizer
from Utility.csv_reader import read_csv
from Utility.Cleaner1 import cleaner1
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from DecisionTree import tree
import pickle

[index,x,y] = read_csv('balanced.csv',3)
sp = int(len(x)/10)
x = x[sp:len(x)]
y = y[sp:len(y)]
x_pred = x[1:sp]
y_test = y[1:sp]


ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
counts = ngram_vectorizer.fit_transform(x)
feature_names = ngram_vectorizer.get_feature_names()

fe = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1),vocabulary=feature_names)
counts_pred = fe.fit_transform(x_pred)

buffer = []
for depth in range(1,260,10):
    decision_tree = tree.decision_tree()
    decision_tree.fit(counts, y, depth)
    y_pred = decision_tree.predict(counts_pred)
    acc = sum(np.transpose(y_pred) == y_test.astype(int))
    buffer.append((depth,acc))

    filename = "depth_{}.csv".format(depth)
pickle.dump(buffer, open("buffer", 'wb'))
df = pd.DataFrame(buffer)
df.to_csv("bufferbuffer.csv",encoding='utf-8',header=['Category'],index_label='Id')


