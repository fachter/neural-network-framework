import numpy as np
import pandas as pd


def get_f1_score(h, y, multiclass=False):
    if (y.ndim > 1 and y.shape[0] > 1) or multiclass:  # the first parameter (row) should show the number of classes
        y_size = y.shape[1]
        h = np.argmax(h, axis=1)
        y = np.argmax(y, axis=1)
        f1_scores = []
        for i in range(y_size):
            f1 = f1_for_class(i, h, y)
            f1_scores.append(f1)

        score_per_class = np.array(f1_scores)
        return score_per_class, score_per_class.mean()
    h = np.round(h)
    f1 = f1_for_class(1, h, y)
    return f1


def f1_for_class(i, h, y):
    true_positives = (h == i) & (y == i)
    false_positives = (h == i) & (y != i)
    false_negatives = (h != i) & (y == i)
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives) + 1e-9)
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives) + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return round(f1, 3)


def get_accuracy(h, y):
    if y.ndim > 1 and y.shape[0] > 1:
        return accuracy_softmax(h, y)
    return accuracy_binary(h, y)


def accuracy_softmax(h, y):
    h = np.argmax(h, axis=1)
    y = np.argmax(y, axis=1)
    df_confusion = pd.crosstab(y, h, rownames=['Actual'], colnames=['Predicted'])
    df_confusion[np.setdiff1d(np.unique(y), np.unique(h))] = 0

    per_class_accuracy = np.diag(df_confusion) / (df_confusion.sum(axis=1) + 1e-9)
    per_class_accuracy = np.array(list(per_class_accuracy))
    return (per_class_accuracy * 100).round(2), (per_class_accuracy.mean() * 100).round(2)


def accuracy_binary(h, y):
    return np.mean(np.rint(h) == y)


def test_for_softmax():
    print(get_f1_score(np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]), np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])))


def test_for_binary():
    print(get_f1_score(
        np.array([
            1, 1, 1, 0, 0, 0
        ]),
        np.array([
            1, 1, 1, 0, 0, 0
        ])
    ))


def test_for_softmax_accuracy():
    print(accuracy_softmax(np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]), np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])))
    print(get_accuracy(np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]), np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])))


def test_for_binary_accuracy():
    print(accuracy_binary(np.array([
        1, 1, 1, 0, 1, 0
    ]), np.array([
        1, 0, 1, 0, 1, 0
    ])))
    print(get_accuracy(np.array([
        1, 1, 1, 0, 1, 0
    ]), np.array([
        1, 0, 1, 0, 1, 0
    ])))


if __name__ == "__main__":
    # f1_score(np.array([0, 0, 0, 1, 1, 1]), np.array([1, 0, 1, 0, 1, 0]))
    # test_for_softmax()
    # test_for_binary()
    # test_for_softmax_accuracy()
    test_for_binary_accuracy()
