import numpy as np

def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE
    # begin answer
    from random import randint
    iteration = 1000000
    n, p = x.shape
    idx = np.zeros((n))
    iter_ctrs = np.zeros((iteration, k, p))
    choosen_ctrs = np.zeros((1, n))
    # initial ctrs
    for i in range(k):
        while True:
            tmp_ctrs = randint(1, n - 1)
            if choosen_ctrs[0, tmp_ctrs] == 1:
                continue
            else:
                choosen_ctrs[0, tmp_ctrs] = 1
                break
        iter_ctrs[0, i, :] = x[tmp_ctrs, :]
    # find best ctrs
    finalIter = 0
    for iter in range(1, iteration):
        cluster_count = np.zeros((1, k))
        for i in range(n):
            current_x = x[i, :]
            current_x_reshape = current_x.reshape(1, -1)
            current_distance_matrix = np.sqrt((abs(iter_ctrs[iter - 1, :, :] - current_x_reshape) ** 2).sum(axis=1, dtype=np.double), dtype=np.double)
            min_distance = current_distance_matrix.min()
            cluster = np.where(current_distance_matrix == min_distance)[0][0]
            idx[i] = int(cluster)
            iter_ctrs[iter, cluster, :] += current_x
            cluster_count[0, cluster] += 1
        for kkk in range(k):
            if cluster_count[0, kkk] == 0:
                cluster_count[0, kkk] = cluster_count[0, kkk] + 0.00001
            iter_ctrs[iter, kkk, :] /= cluster_count[0, kkk]
        dis = iter_ctrs[iter, :, :] - iter_ctrs[iter - 1, :, :]
        if np.count_nonzero(dis) == 0:
            finalIter = iter
            break
    ctrs = iter_ctrs[finalIter - 1, :, :]
    iter_ctrs = iter_ctrs[0: finalIter - 1, :, :]
    return idx, ctrs, iter_ctrs
