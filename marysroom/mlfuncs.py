import numpy as np


def one_hot(y_train, n_outs):
    one_hot = np.zeros([y_train.shape[0], n_outs])
    one_hot[range(y_train.shape[0]), y_train] = 1
    return one_hot
