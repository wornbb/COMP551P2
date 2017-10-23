
from sklearn.feature_extraction.text import TfidfVectorizer
from Utility.csv_reader import read_csv
from Utility.Cleaner1 import cleaner1
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from DecisionTree import tree
import pickle

[index,x,y] = read_csv('balanced.csv',3)
[index,x_pred] = read_csv('test_set_x.csv')
x = x[1:len(x)]
y = y[1:len(y)]
x_pred = x_pred[1:len(x_pred)]

# vectorizer for the training & testing sets
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
counts = ngram_vectorizer.fit_transform(x)
feature_names = ngram_vectorizer.get_feature_names()
fe = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1),vocabulary=feature_names)
counts_pred = fe.fit_transform(x_pred)
# train the model
decision_tree = tree.decision_tree()
decision_tree.fit(counts,y,51)
# save to file
filename = 'depth_31_model.sav'
pickle.dump(decision_tree, open(filename, 'wb'))

# make prediction and save to file
y_pred = decision_tree.predict(counts_pred)
df = pd.DataFrame(y_pred.astype(np.int32))
df.to_csv("predictions.csv",encoding='utf-8',header=['Category'],index_label='Id')

