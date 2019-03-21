import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    # begin answer
    # 对矩阵X增广
    new_X = np.concatenate((np.ones((1, N)), X), axis=0)
    # 获得新的矩阵的rows columns
    new_P, new_N = new_X.shape
    while True:
        iters += 1
        # every sample sigma, row vector
        sigma_matrix = np.matmul(w.T, new_X)
        activation_matrix = np.sign(sigma_matrix)
        activation_matrix_idx = np.argwhere(activation_matrix == -1.)
        discriminant_matrix = np.multiply(np.matmul(w.T, new_X), y)
        discriminant_matrix_idx = np.argwhere(discriminant_matrix <= -0.)
        if discriminant_matrix_idx.shape[0] == 0:
            break
        error_matrix_X = new_X[:, discriminant_matrix_idx[:, 1]]
        error_matrix_y = y[:, discriminant_matrix_idx[:, 1]]
        temp = np.sum(np.multiply(error_matrix_X, error_matrix_y), axis=1).reshape(-1, 1)
        w = w + temp
    # end answer
    return w, iters

