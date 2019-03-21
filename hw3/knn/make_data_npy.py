import os
import numpy as np
from extract_image import extract_image
from show_image import show_image

path = "./download/res"
path_test = "./download/res/0.gif"
test = extract_image(path_test)
num_of_image, num_of_input = test.shape
features = np.zeros((num_of_image * 1000, num_of_input))
label = np.zeros((num_of_image * 1000, 1))
i = 0
for file in os.listdir(path):
    full_path = os.path.join(path, file)
    s_x = extract_image(full_path)
    show_image(s_x)
    s_number = input("Please input the label\n")
    if s_number[0] == 'q':
        break
    features[i:i+5, :] = s_x
    for j in range(5):
        label[i + j, 0] = int(s_number[j])
    i = i + 5

np.savez("hack_data.npz", x_train=features, y_train=label)