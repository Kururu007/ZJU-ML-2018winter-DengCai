import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    # mean = np.mean(data, axis=0)
    # data = data - mean
    covData = np.cov(data.T)
    eigvalue, eigvector = np.linalg.eig(covData)
    new_idx = np.argsort(-eigvalue)
    eigvector = eigvector[:, new_idx]
    return eigvector, eigvalue
    # end answer

