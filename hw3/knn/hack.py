import numpy as np

from knn import knn
import show_image
from extract_image import extract_image

def hack(img_name):
    '''
    HACK Recognize a CAPTCHA image
      Inputs:
          img_name: filename of image
      Outputs:
          digits: 1x5 matrix, 5 digits in the input CAPTCHA image.
    '''
    data = np.load('hack_data.npz')

    # YOUR CODE HERE (you can delete the following code as you wish)
    x_train = data['x_train']
    y_train = data['y_train']

    # begin answer
    x = extract_image(img_name)
    digits = knn(x, x_train, y_train, 10)
    # end answer

    return digits
