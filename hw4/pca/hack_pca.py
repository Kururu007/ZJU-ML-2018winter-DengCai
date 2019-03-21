import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)
    # YOUR CODE HERE
    # begin answer
    img_L = rgb2gray(img_r)
    x_idx, y_idx = np.where(img_L < 30)
    input_data = np.hstack((x_idx.reshape(-1, 1), y_idx.reshape(-1, 1)))
    mean = np.mean(input_data, axis=0)
    input_data = input_data - mean
    from pca import PCA
    eigvector, eigvalue = PCA(input_data)
    vec = eigvector[:, 0]
    angle = np.arctan(vec[0]/vec[1]) * 180 / np.pi
    from scipy.ndimage import rotate
    final_image = rotate(img_r, -angle).astype(np.int)
    return final_image



    # end answer