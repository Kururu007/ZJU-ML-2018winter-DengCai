import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE
    # begin answer
    N_test = x.shape[0]
    N = x_train.shape[0]
    y = np.zeros((N_test, 1))
    distance_ = np.zeros((N_test, N))
    for i in range(N_test):
        for j in range(N):
            distance_[i, j] = np.linalg.norm(x[i, :] - x_train[j, :])
    for i in range(N_test):
        distance_s = distance_[i, :]
        index = np.argsort(distance_s, )
        finding = y_train[index[0:k]]
        fre_mode, fre_count = scipy.stats.mode(finding)
        idx = np.where(fre_count == fre_count.max())
        y[i, 0] = fre_mode[idx[0]]
    # end answer

    return y
