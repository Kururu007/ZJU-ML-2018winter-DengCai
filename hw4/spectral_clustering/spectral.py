import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    n = W.shape[0]
    D = np.diagflat(np.sum(W, axis=0).reshape(1, -1).flatten())
    L = D - W
    Dm = np.diagflat(1. / np.sqrt(np.sum(W, axis=0).reshape(1, -1)).flatten())
    L = np.matmul(np.matmul(Dm, L), Dm)
    w, v = np.linalg.eig(L)
    index = np.argsort(w)[0:k]
    v_k = v[:, index]
    v_k = np.matmul(np.sqrt(D), v_k)
    idx = kmeans(v_k, k)
    return idx
    # end answer
