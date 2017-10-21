
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from Utility.csv_reader import read_csv
from Utility.Cleaner1 import cleaner1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white")


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
logistic = LogisticRegression()
logistic.fit(counts,y)
y_pred = logistic.predict(counts_pred)
df = pd.DataFrame(y_pred)
df.to_csv("predictions.csv",encoding='utf-8',header=['Category'],index_label='Id')

x_min,x_max = counts.min()- 0.5, counts.max() + 0.5
#y_min,y_max = counts[:,1].min()- 0.5, counts[:,1].max() + 0.5
xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
test2 = np.array([[1,2,3]]).transpose()
grid = np.tile(np.c_[xx.ravel()],(1,len(feature_names)))
#grid = np.c_[xx.ravel(),yy.ravel()]

probs = logistic.predict_proba(grid)[:,1].reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)


ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")
plt.show(f)