from Utility.csv_reader import read_csv
from Utility.Cleaner1 import cleaner1
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from DecisionTree import tree
import pickle

# Create a balanced input
[index,x,y] = read_csv('cleaner1.csv',3)
x = x[1:len(x)]
y = y[1:len(y)]
unique, counts = np.unique(y,return_counts = True)
max_sample_per_class = counts.min()
table = np.column_stack((x,y))
balanced_table = np.array([])
for label in unique:
    index = (y==label).nonzero()[0]
    balanced_x = x.take(index,axis=0)[0:max_sample_per_class]
    balanced_y = y.take(index,axis=0)[0:max_sample_per_class]
    add_table = np.column_stack((balanced_x,balanced_y))
    if balanced_table.size:
        balanced_table = np.vstack((balanced_table,add_table))
    else:
        balanced_table = add_table
df = pd.DataFrame(balanced_table)
df.to_csv("balanced.csv",encoding='utf-8')
