import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    P = np.zeros((N, K))
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    for i in range(0, N):
        x = X[:, i]
        # print(x, type(x), x.shape)
        
        for j in range(0, K):
            begin_idx = j * K
            end_idx = j * K + 2
            # print(begin_idx, end_idx)
            u = Mu[begin_idx:end_idx, ]
            # print(u, u.shape)
            co_variance = Sigma[:, :, j]
            # print(co_variance)
            # print((x - u).shape)
            substract = x - u
            substract = substract.reshape(-1, 1)
            substract_t = np.transpose(substract)
            # print(substract.shape, substract_t.shape, np.linalg.inv(co_variance).shape)
            value1 = (1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(co_variance))))
            value2 = np.exp(-0.5 * np.matmul(np.matmul(substract_t, np.linalg.inv(co_variance)), substract))
            P[i, j] = value1 * value2[0][0]
            

    for i in range(0, N):
        p_X = 0.0
        for t in range(0, K):
            p_X = p_X + Phi[t] * P[i, t]
        for k in range(0, K):
            p[i, k] = Phi[k] * P[i, k] / p_X

            # print(value1, type(value1)) 
            # print(value2, type(value2)) 
    # end answer
    
    return p
    