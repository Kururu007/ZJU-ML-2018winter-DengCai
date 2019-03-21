import numpy as np

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0
    from sklearn.metrics.pairwise import linear_kernel
    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    # before class following
    from scipy.optimize import minimize
    new_X = np.concatenate((np.ones((1, N)), X), axis=0)
    C = 20
    def loss_f(w):
        return np.sum(np.maximum(0, 1 - np.multiply(y, np.matmul(w.T, new_X)))) + 0.5 / C * np.sum(np.multiply(w, w))
    solution = minimize(loss_f, w, method='SLSQP')
    w = solution.x
    wx_b = np.matmul(w.T, new_X)
    num = np.where(np.logical_and(wx_b > -1, wx_b < 1))[0].shape[0]
    # end answer
    return w, num

