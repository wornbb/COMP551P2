"""This script is used to clean the training set with the following sequence of operations:
        1. All non-language symbols(question marks, emojis, numbers) are removed
        2. All entries with less than 5 chars remaining will be removed
        3. All entries with key words 'http', 'php' are removed
        4. """
import numpy as np
import pandas as pd
from Utility.csv_reader import read_csv
def cleaner1(mode,*files):
    del_items = []
    [index,x] = read_csv(files[0])
    try:
        [index,y] = read_csv(files[1])
        existence_of_y = 1
    except:
        existence_of_y = 0
    k = np.nditer(x,flags=['f_index'])
    while not k.finished:
        entry = x[k.index]
        entry = ''.join(filter(str.isalpha,entry))
        if len(entry)>=5:
            if not (('http' in entry) or ('php' in entry)):
                x[k.index] = entry
            else:
                del_items.append(k.index)
        else:
            del_items.append(k.index)
        k.iternext()
    mask = np.ones(len(x),dtype=bool)
    mask[del_items] = False
    x = x[mask]
    y = y[mask]
    index = index[mask]
    return [index,x,y]
    #for pointer,entry,label in zip(index,x,y):

train_set_x = 'train_set_x.csv'
train_set_y = 'train_set_y.csv'
[index,x,y] = cleaner1("train",train_set_x,train_set_y)
table = np.column_stack((x,y))
df = pd.DataFrame(table)
df.to_csv("test_utf8.csv",encoding='utf-8')
#[index,x] = read_csv(train_set_x)
#[index,y] = read_csv(train_set_y)
#table = np.column_stack((index,x,y))

