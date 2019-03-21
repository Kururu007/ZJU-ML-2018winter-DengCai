import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    n, p = X.shape
    W = np.zeros((n, n))
    Distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(X[i, :] - X[j, :])
            Distance_matrix[i, j] = distance
            Distance_matrix[j, i] = distance
    # Distance_matrix = (X, X)
    dis = np.sort(Distance_matrix, axis=1)
    index = np.argsort(Distance_matrix, axis=1)
    K_neighbors = dis[:, 0:k]
    K_neighbors[K_neighbors > threshold] = 0
    K_neighbors[K_neighbors > 0] = 1
    for i in range(n):
        W[i, index[i, 0:k]] = K_neighbors[i, :]
        W[index[i, 0:k], i] = K_neighbors[i, :].T
    # end answer
    return W
