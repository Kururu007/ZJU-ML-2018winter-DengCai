import numpy as np

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Liklihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    '''

    C, N = x.shape
    l = np.zeros((C, N))
    #TODO

    # begin answer
    row_sum = np.sum(x, axis=1)
    print(row_sum)
    
    for i in range(0, C):
        for j in range(0, N):
            l[i, j]= (x[i,j]/row_sum[i])
    # end answer

    return l