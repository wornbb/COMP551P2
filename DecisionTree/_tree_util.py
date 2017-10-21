import numpy as np
import math
def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}
def split(x,y,thres):
    x = np.double(x)
    y = np.array(y)
    index_left = (x < thres).nonzero()[0]
    index_right = (x >= thres).nonzero()[0]
    y_left = y.take(index_left, axis=0)
    y_right = y.take(index_right, axis=0)
    x_left = x.take(index_left, axis=0)
    x_right = x.take(index_right, axis=0)
    return np.array([x_left,y_left],[x_right,y_right])
def optimal_thres(x,y):
    ig_record = [[candidate_split_value,ig_split(x,y,candidate_split_value)] for candidate_split_value in np.unique(x)]
    best_ig = max(np.array(ig_record)[:,1])
    best_split_value = [candidate_split_value for candidate_split_value,ig_value
                        in ig_record if np.isclose(ig_value ,best_ig,rtol=1e-6)]
    return best_split_value
def entropy(x):
    res = 0
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res
def ig_split(x,y,thres):
    x = np.double(x)
    y = np.array(y)
    index_left = (x<thres).nonzero()[0]
    index_right = (x>=thres).nonzero()[0]
    p_left = len(index_left)/len(x)
    p_right = len(index_right)/len(x)
    ref_entropy = entropy(y)
    y_left = y.take(index_left,axis=0)
    y_right = y.take(index_right,axis=0)
    left_entropy = p_left*entropy(y_left)
    right_entropy = p_right * entropy(y_right)
    ig = ref_entropy-left_entropy-right_entropy
    return ig

def ig_freature(x, y):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def spliter(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([ig_freature(x_attr, y) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y
    # We split using the selected attribute
    thres = optimal_thres(x,y)
    sets = split(x,y,thres)
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (selected_attr, k)] = spliter(x_subset, y_subset)

    return res
