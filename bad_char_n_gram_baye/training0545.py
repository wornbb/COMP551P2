
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from Utility.csv_reader import read_csv
from Utility.Cleaner1 import cleaner1
import pandas as pd
[index,x,y] = read_csv('cleaner1.csv',3)
[index,x_pred] = read_csv('test_set_x.csv')
x = x[1:len(x)]
y = y[1:len(y)]
x_pred = x_pred[1:len(x_pred)]

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
counts = ngram_vectorizer.fit_transform(x)
feature_names = ngram_vectorizer.get_feature_names()


fe = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1),vocabulary=feature_names)
counts_pred = fe.fit_transform(x_pred)
mnb = MultinomialNB()
mnb.fit(counts,y)
y_pred = mnb.predict(counts_pred)
df = pd.DataFrame(y_pred)
df.to_csv("predictions.csv",encoding='utf-8',header=['Category'],index_label='Id')
