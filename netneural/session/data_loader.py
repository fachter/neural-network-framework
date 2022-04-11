import random
from math import ceil

import numpy as np


def train_test_split(X, y, train_per, test_per, valid_per=None, randomized=True):
    """
    Randomized training test split. If valid_per parameter is given, a validation set is output as well.
    :param randomized: signals if data should be sampled randomized or not, default is True
    :param X: dataset
    :param y: validation data
    :param train_per: desired percentage for training data
    :param test_per: desired percentage for test data
    :param valid_per: desired percentage for validation data, default is 0
    :return: X_train, y_train, X_test, y_test, X_valid, y_valid
    """
    if valid_per is None:
        assert train_per + test_per == 1
        valid_per = 0
    else:
        assert train_per + test_per + valid_per == 1

    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()
    X_valid = list()
    y_valid = list()

    if len(y.shape) == 1:  # only one dimensional
        y = y.reshape(1, len(y))  # turn two dimensional

    if randomized:
        for idx in range(len(X)):
            n = random.choices([1,2,3], [train_per, test_per, valid_per])[0]
            if n == 1:
                X_train.append(X[idx])
                y_train.append(y[:,idx])
            elif n == 2:
                X_test.append(X[idx])
                y_test.append(y[:,idx])
            elif n == 3:
                X_valid.append(X[idx])
                y_valid.append(y[:,idx])
    else:
        training_idx = int(train_per*len(X))
        X_train = X[:training_idx]
        y_train = y[:training_idx]
        if valid_per == 0:
            X_test = X[training_idx:]
            y_test = y[training_idx:]
        else:
            test_idx = int(test_per*len(X))
            X_test = X[training_idx:(training_idx+test_idx)]
            y_test = y[training_idx:(training_idx+test_idx)]
            X_valid = X[(training_idx+test_idx):]
            y_valid = y[(training_idx+test_idx):]

    print('New data distribution: \n' + str(len(X_train)) + ' trainings samples \n' + str(len(X_test)) + ' test samples'
                                '\n' + str(len(X_valid)) + ' validation samples\n')
    return np.array(X_train), np.array(y_train).T, np.array(X_test), np.array(y_test).T, np.array(X_valid), np.array(y_valid).T


def train_test_split_idxs(n_data, train_per):
    """
    Randomized training test split. If valid_per parameter is given, a validation set is output as well. Instead of the
    actual data, returns list of indices that can be used for slicing the datasets.
    :param n_data: number of data points to split
    :param train_per: desired percentage for training data
    :return: List of indices for X_train, y_train, X_test, y_test, X_valid, y_valid
    """
    n_train = ceil(train_per * n_data)
    idxs = list(range(n_data))
    random.shuffle(idxs)

    print('New data distribution: \n' + str(n_train) + ' trainings samples \n' + str(n_data-n_train) + ' test samples\n')
    return idxs[:n_train], idxs[n_train:]

if __name__ == '__main__':
    #file_path = '../data/data_merge/output_csv/rotate_felix_gt.csv'
    #raw_data = pd.read_csv(file_path)
    #X, y = preprocess_data(raw_data, including_ground_truth=True)
    #with open('test.npy', 'wb') as f:
        #np.save(f, X)
        #np.save(f, y)
    with open('../neural_net/test.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
    X_train, y_train, X_test, y_test, X_valid, y_valid = train_test_split(X, y, train_per=0.8, test_per=0.2, randomized=False)