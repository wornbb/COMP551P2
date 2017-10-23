from collections import Counter

import numpy as np
from DecisionTree.util._tree_util import ig_freature
from DecisionTree.util._tree_util import is_pure
from DecisionTree.util._tree_util import optimal_thres
from DecisionTree.util._tree_util import split
from DecisionTree.util._tree_util import ig_split

class node(): # node of decision tree
    def __init__(self):
        self.split_atrr = None
        self.split_thres = None
        self.left_child = None
        self.right_child = None
    def get_name(self):
        return 'Node'
class leaf(): #leaf of decision tree
    def __init__(self):
        self.label = None
    def get_name(self):
        return 'leaf'

class decision_tree():
    def __init__(self):
        self.root = None #decision tree
        self.depth = None # hyperparameter
    def fit(self, x, y, depth):
        '''
        :param x: feature set, each column represent a different feature
        :param y: label vector
        :param depth: hyperparameter
        :return:
        '''
        self.depth = depth
        x = x.toarray().astype(np.int32)  # n-gram should yield integer feature values
        y = np.array([int(i) for i in y])
        current_deppth = 0
        self.root = self.spliter(x,y, current_deppth) #recursively build the tree
        return 0

    def predict(self,x):
        '''
        :param x: feature set, each column represent a different feature
        :return:
        prediction 2d np.array, vector
        '''
        x = x.toarray().astype(np.int32)  # n-gram should yield integer feature values
        try: # only used for hand-made validation.
            samples = x.shape[0] #when x is np.array, this is the case for our project
        except:
            samples = len(x) # when x is a list, this is the case when testing the codes during the development
        y = np.empty([samples,1])
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


    def spliter(self,x, y, current_depth):
        '''
        spliter used to build the tree
        :param x: feature set, each column represent a different feature
        :param y: label vector
        :param depth: hyperparameter
        :return:
        '''
        # If there could be no split, just return the original set
        tree = node()
        if current_depth>=self.depth: #exit condition, when maximum depth reached
            tree = leaf()
            tree.label, _ = Counter(y).most_common(1)[0]
            return  tree
        if is_pure(y) or len(y) == 0: #exit condition when no need to further split
            tree = leaf()
            tree.label, _ = Counter(y).most_common(1)[0]
            return tree
        # We get attribute that gives the highest information gain
        gain = np.array([ig_freature(x_attr, y) for x_attr in x.T])
        selected_attr = np.argmax(gain)
        # If there's no gain at all, nothing has to be done, just return the original set
        if np.all(gain < 1e-6): #exit condition when no need to further split
            tree = leaf()
            tree.label,_ = Counter(y).most_common(1)[0]
            return tree
        # We split using the selected attribute
        thres = optimal_thres(x[:, selected_attr], y)

        # special case handler. When decision tree is at its root
        if tree.split_thres == None:
            tree.split_thres = thres
            tree.split_atrr =selected_attr
        else:pass
        sets = split(x, y, thres, selected_attr)
        # recursively grow the tree
        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)
            if k==0:
                #this goes left
                tree.left_child = self.spliter(x_subset,y_subset,current_depth+1)
            elif k==1:
                # this goes right
                tree.right_child = self.spliter(x_subset,y_subset,current_depth+1)
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
