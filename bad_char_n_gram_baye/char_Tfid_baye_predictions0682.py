
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from Utility.csv_reader import read_csv
from Utility.Cleaner1 import cleaner1
import pandas as pd
[index,x,y] = read_csv('cleaner1.csv',3)
[index,x_pred] = read_csv('test_set_x.csv')
x = x[1:len(x)]
y = y[1:len(y)]
x_pred = x_pred[1:len(x_pred)]

# vectorizer for the training & testing sets
Tfid_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 1))
counts = Tfid_vectorizer.fit_transform(x)
feature_names = Tfid_vectorizer.get_feature_names()
fe = TfidfVectorizer(analyzer='char', ngram_range=(1, 1),vocabulary=feature_names)

counts_pred = fe.fit_transform(x_pred)
mnb = MultinomialNB()
mnb.fit(counts,y)
y_pred = mnb.predict(counts_pred)
df = pd.DataFrame(y_pred)
df.to_csv("predictions.csv",encoding='utf-8',header=['Category'],index_label='Id')
