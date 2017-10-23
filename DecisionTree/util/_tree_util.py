import numpy as np
import math
def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}
def split(x,y,thres,x_attr):

    x_selected = list(x[:,x_attr])
    index_left = (x_selected <= thres).nonzero()[0]
    index_right = (x_selected >thres).nonzero()[0]
    return {0:index_left,1:index_right}

def optimal_thres(x,y):
    ig_record = [[candidate_split_value, ig_split(x, y, candidate_split_value)] for candidate_split_value in np.unique(x)]
    best_ig = max(np.array(ig_record)[:, 1])
    best_split_value = [candidate_split_value for candidate_split_value,ig_value
                        in ig_record if np.isclose(ig_value ,best_ig,rtol=1e-6)]
    return best_split_value[0]
def entropy(x):
    res = 0
    val, counts = np.unique(x, return_counts=True)
    try: samples = x.shape[0]
    except: samples = len(x)
    freqs = counts.astype('float')/samples
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res
def ig_split(x,y,thres):
    #x = x.toarray().astype(np.int32)
    #y = np.array([int(i) for i in y])
    index_left = (x<=thres).nonzero()[0]
    index_right = (x>thres).nonzero()[0]
    try: samples = x.shape[0]
    except: samples = len(x)
    p_left = len(index_left)/samples
    p_right = len(index_right)/samples
    ref_entropy = entropy(y)
    y_left = y.take(index_left,axis=0)
    y_right = y.take(index_right,axis=0)
    left_entropy = p_left*entropy(y_left)
    right_entropy = p_right * entropy(y_right)
    ig = ref_entropy-left_entropy-right_entropy
    return ig

def ig_freature(x, y):
    #x = x.toarray().ravel().astype(np.int32)
    #y = np.array([int(i) for i in y])
    res = entropy(y)
    try: samples = x.shape[0]
    except: samples = len(x)
    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/samples

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x==v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def spliter(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y[0]

    # We get attribute that gives the highest mutual information
    gain = np.array([ig_freature(x_attr, y) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y
    # We split using the selected attribute
    thres = optimal_thres(x[:,selected_attr],y)
    sets = split(x,y,thres,selected_attr)
    #sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["%d, %d" % (selected_attr, thres)] = spliter(x_subset, y_subset)
        print(res)
    return res

if __name__ == '__main__':
    x1 = [0, 1, 1, 2, 2, 2]
    x2 = [0, 0, 1, 1, 1, 0]
    X = np.array([x1, x2]).T
    y = np.array([1, 0, 0, 1, 1, 0])
    a = spliter(X,y)
    print(a)