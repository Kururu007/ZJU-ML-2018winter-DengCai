import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    new_X = np.concatenate((np.ones((1, N)), X), axis=0)
    y[y == -1] = 0
    lr = 1e-4
    iter_num = 100
    for i in range(iter_num):
        predictions = sigmoid(np.matmul(w.T, new_X))
        gradient = np.matmul(y - predictions, new_X.T).T
        w = w + lr * (gradient - 2 * lmbda * w)
    y[y == 0] = -1.
    # end answer
    return w
