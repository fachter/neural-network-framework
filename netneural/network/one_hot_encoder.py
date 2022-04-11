import numpy as np


class OneHotEncoder:

    unique_labels = None

    def __init__(self, unique_labels=None):
        self.unique_labels = unique_labels
        if self.unique_labels is not None:
            np.sort(self.unique_labels)

    def encode(self, x):
        # returns alphabetically sorted list of unique elements
        if self.unique_labels is None:
            self.unique_labels, inverse = np.unique(x, return_inverse=True)
        else:
            sorter = np.argsort(self.unique_labels)
            inverse = sorter[np.searchsorted(self.unique_labels, x, sorter=sorter)]
        eye = np.eye(self.unique_labels.shape[0])
        result = eye[inverse]
        return result

    def decode(self, x):
        arg_max = np.argmax(x, axis=1)
        arg_max_int = arg_max.astype(int)
        return np.array(self.unique_labels)[arg_max_int]
