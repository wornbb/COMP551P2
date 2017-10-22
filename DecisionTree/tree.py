from collections import Counter

import numpy as np
from DecisionTree.util._tree_util import ig_freature
from DecisionTree.util._tree_util import is_pure
from DecisionTree.util._tree_util import optimal_thres
from DecisionTree.util._tree_util import split

x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
x3 = [0, 2, 0, 1, 1, 0]
y = np.array([0, 2, 0, 1, 1, 0])
class node():
    def __init__(self):
        self.split_atrr = None
        self.split_thres = None
        self.left_child = None
        self.right_child = None
    def get_name(self):
        return 'Node'
class leaf():
    def __init__(self):
        self.label = None
    def get_name(self):
        return 'leaf'

class decision_tree():
    def __init__(self):
        self.root = None
    def fit(self, x, y):
        #self.build_root(x,y)
        self.root = self.spliter(x,y)
        return 0

    def predict(self,x):
        x = np.array(x)
        y = np.empty([len(x),1])
        index = 0
        for test in x:
            tree = self.root
            while tree.get_name() != 'leaf':
                selected_attr = tree.split_atrr
                thres = tree.split_thres
                if test[selected_attr] <= thres:
                    tree = tree.left_child
                else:
                    tree = tree.right_child
            y[index,0] = tree.label
            index += 1
        return y



    def spliter(self,x, y):
        # If there could be no split, just return the original set
        tree = node()
        if is_pure(y) or len(y) == 0:
            tree = leaf()
            #tree.label = np.bincount(y).argmax()
            tree.label, _ = Counter(y).most_common(1)[0]
            return tree
        # We get attribute that gives the highest mutual information
        gain = np.array([ig_freature(x_attr, y) for x_attr in x.T])
        selected_attr = np.argmax(gain)
        # If there's no gain at all, nothing has to be done, just return the original set
        if np.all(gain < 1e-6):
            tree = leaf()
            tree.label,_ = Counter(y).most_common(1)[0]
            return tree
        # We split using the selected attribute
        thres = optimal_thres(x[:, selected_attr], y)
        if tree.split_thres == None:
            tree.split_thres = thres
            tree.split_atrr =selected_attr
        else:pass
        sets = split(x, y, thres, selected_attr)
        # sets = partition(x[:, selected_attr])
        #res = {}
        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)
            if k==0:
                #this goes left
                tree.left_child = self.spliter(x_subset,y_subset)
            elif k==1:
                tree.right_child = self.spliter(x_subset,y_subset)
                #this goes right
          #  res["%d, %d" % (selected_attr, thres)] = spliter(x_subset, y_subset)
        return tree

if __name__ == '__main__':
    x1 = [0, 1, 1, 2, 2, 2]
    x2 = [0, 0, 1, 1, 1, 0]
    x = np.array([x1,x2]).T
    y = np.array([1, 0, 0, 1, 1, 0])
    a = decision_tree()
    a.fit(x,y)
    y = a.predict(x)
    print(y)
