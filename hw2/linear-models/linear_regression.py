import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    # 对矩阵X增广
    X = np.concatenate((np.ones((1, N)), X), axis=0)
    # 获得新的矩阵的rows columns
    # new_P, new_N = new_X.shape
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X, X.T)), X), y.T)
    # end answer
    return w
