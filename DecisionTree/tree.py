import numpy as np
from ._tree_util import partition
from ._tree_util import entropy
from ._tree_util import is_pure
from ._tree_util import spliter

x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
x3 = [0, 2, 0, 1, 1, 0]
y = np.array([0, 2, 0, 1, 1, 0])
class tree:
    def __init__(self,):
        self.rules
    def fit(self, x, y):
        res = spliter(x,y)