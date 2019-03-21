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

if __name__ == "__main__":
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt

    data = sio.loadmat('data.mat')
    x1_train, x1_test, x2_train, x2_test = data['x1_train'], data['x1_test'], data['x2_train'], data['x2_test']
    all_x = np.concatenate([x1_train, x1_test, x2_train, x2_test], 1)
    data_range = [np.min(all_x), np.max(all_x)]

    from get_x_distribution import get_x_distribution

    train_x = get_x_distribution(x1_train, x2_train, data_range)
    test_x = get_x_distribution(x1_test, x2_test, data_range)
    from likelihood import likelihood

    l = likelihood(train_x)
    print(l)
    width = 0.35
    p1 = plt.bar(np.arange(data_range[0], data_range[1] + 1), l.T[:,0], width)
    p2 = plt.bar(np.arange(data_range[0], data_range[1] + 1) + width, l.T[:,1], width)
    plt.xlabel('x')
    plt.ylabel('$P(x|\omega)$')
    plt.legend((p1[0], p2[0]), ('$\omega_1$', '$\omega_2$'))
    plt.axis([data_range[0] - 1, data_range[1] + 1, 0, 0.5])
    plt.show()