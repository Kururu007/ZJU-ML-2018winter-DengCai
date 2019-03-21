import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    new_X = np.concatenate((np.ones((1, N)), X), axis=0)
    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(new_X, new_X.T) + lmbda * np.eye(P + 1)), new_X), y.T)
    # end answer
    return w
