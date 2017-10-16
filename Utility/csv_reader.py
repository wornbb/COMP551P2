import  csv
import numpy as np
def read_csv(file_name):
    """ This function read the dataset in format given by the instructor
        The result should be accessed from 1. For example, the first index should be index[1] NOT index[0]
        since the index[0] is actually the name of that column in the csv file.
    """
    with open(file_name, 'r', encoding="utf-8") as csvfile:
        train_set_x = csv.reader(csvfile, delimiter=',', quotechar='|')
        index, text = zip(*((c[0], c[1]) for c in train_set_x))
        # Potentially this could be changed to a 2D numpy array directly, like a table.
        # But at the time writing this function, which way is more convenient is unknown
    return [np.array(index),np.array(text)]