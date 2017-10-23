
from Utility.csv_reader import read_csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from DecisionTree import tree
import pickle
# load the balanced training set
[index,x,y] = read_csv('balanced.csv',3)

# split the balanced training set into cross-validation training set and testing set
sp = int(len(x)/10)
l = len(x)
temp = np.column_stack((x,y))
np.random.shuffle(temp)
# for training
x = temp[sp:len(x),0]
y = temp[sp:len(y),1]
# for predicting
x_pred = temp[1:sp,0]
y_test = temp[1:sp,1]

# 1-gram feature extraction
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
# extraction for training set
counts = ngram_vectorizer.fit_transform(x)
feature_names = ngram_vectorizer.get_feature_names()
# extraction for testing set
fe = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1),vocabulary=feature_names)
counts_pred = fe.fit_transform(x_pred)

buffer = []
#for depth in range(1,260,10): #use this line for coarse tuning
for depth in range(12,15):
    decision_tree = tree.decision_tree()
    decision_tree.fit(counts, y, depth)
    y_pred = decision_tree.predict(counts_pred)
    acc = sum((np.transpose(y_pred) == y_test.astype(int))[0,:])
    buffer.append((depth,acc))

    filename = "depth_{}".format(depth)
    pickle.dump(decision_tree, open(filename, 'wb'))
    print(depth, acc) # used for progress tracking
pickle.dump(buffer, open("buffer", 'wb'))
df = pd.DataFrame(buffer)
#df.to_csv("bufferbuffer.csv",encoding='utf-8',header=['Category'],index_label='Id')


