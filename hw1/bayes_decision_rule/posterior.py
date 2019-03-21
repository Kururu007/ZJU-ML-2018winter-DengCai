import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO

    # begin answer
    # total is the sum of the original dataset, though the input x is the distribution of dataset
    # total_class is the sum of every class from the original dataset
    total_class = x.sum(axis=1)
    p_w = total_class/total
    for i in range(0, C):
        for j in range(0, N):
            p[i, j] = p_w[i] * l[i, j] / (l[0, j] * p_w[0] + l[1, j] * p_w[1]);
    # end answer
    
    return p
