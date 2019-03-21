import numpy as np
import matplotlib.pyplot as plt


def kmeans_plot(x, idx, ctrs, iter_ctrs):
    """
    Input:  x - data point features, n-by-p maxtirx.
            idx  - cluster label
            ctrs - cluster centers, K-by-p matrix.
            iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    """
    plt.figure(figsize=(10, 10))

    color = ['red', 'blue']
    fmt = ['rs-', 'bo-']
    for label in np.unique(idx):
        plt.scatter(x[np.where(idx == label)[0], 0], x[np.where(idx == label)[0], 1], s=3, c=color[int(label)])
        plt.plot(iter_ctrs[:, int(label), 0], iter_ctrs[:, int(label), 1],
                 fmt[int(label)], linewidth=2, markersize=5)
