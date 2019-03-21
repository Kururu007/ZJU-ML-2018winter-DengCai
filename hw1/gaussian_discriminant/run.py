#!/usr/bin/env python
# coding: utf-8

# # Gaussian Discriminant Analysis and MLE
# *Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. Please check the pdf file for more details.*
# 
# In this exercise you will:
#     
# - implement the calculation of **gaussian posterior probability** of GDA
# - find appropriate tuples of parameters for variout kinds of **decision boundary**

# In[1]:


# some basic imports
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
import math
from plot_ex1 import plot_ex1, figure


# ### Decision boundaries are plotted in the figure variable. You may need to refer to it after every plot

# In[2]:


# mu: 2x1 matrix, e.g. np.array([[0,0]]).T
# Sigma: 2x2 matrix e.g. nnp.array([[1, 0], [0, 1]]).T
# phi: a number e.g 0.5

# change the value for specific decision boundary

# begin answer
mu0 = np.array([0, 0])
Sigma0 = np.array([[1, 0], [0, 1]]);
mu1 = np.array([1, 1]);
Sigma1 = np.array([[1, 0], [0, 1]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line', 1)
print("finished!")
# figure


# In[3]:


# begin answer
mu0 = np.array([5, 0])
Sigma0 = np.array([[5, 0], [0, 11]]);
mu1 = np.array([5, 9]);
Sigma1 = np.array([[5, 0], [0, 11]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line (one side)', 2)
# figure
print("finished!")

# In[ ]:


# begin answer
mu0 = np.array([0, 0])
Sigma0 = np.array([[2, 0], [0, 2]]);
mu1 = np.array([2, 2]);
Sigma1 = np.array([[4, 0], [0, 1]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Parabalic', 3)
# figure
print("finished!")

# In[ ]:


# begin answer
mu0 = np.array([3, 3])
Sigma0 = np.array([[1, 0], [0, 7]]);
mu1 = np.array([3, 3]);
Sigma1 = np.array([[4, 0], [0, 1]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Hyperbola', 4)
# figure
print("finished!")

# In[ ]:


# begin answer
mu0 = np.array([1, 1])
Sigma0 = np.array([[6, 0], [0, 6]]);
mu1 = np.array([1, 1]);
Sigma1 = np.array([[1, 0], [0, 6]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Two parallel lines.', 5)
# figure
print("finished!")

# In[ ]:


# begin answer
mu0 = np.array([15, 15])
Sigma0 = np.array([[1, 0], [0, 1]]);
mu1 = np.array([15, 15]);
Sigma1 = np.array([[2, 0], [0, 2]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Circle', 6)
# figure
print("finished!")

# In[ ]:


# begin answer
mu0 = np.array([9, 9])
Sigma0 = np.array([[1, 0], [0, 1]]);
mu1 = np.array([9, 9]);
Sigma1 = np.array([[15, 0], [0, 6]]);
phi = 0.5;
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Ellipsoid', 7)
# figure
print("finished!")

# In[ ]:


# begin answer
mu0 = np.array([0, 0])
Sigma0 = np.array([[1, 0], [0, 1]]);
mu1 = np.array([0, 0]);
Sigma1 = np.array([[1, 0], [0, 1]]);
phi = 0.5;
# end answer


plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'No boundary', 8)
# figure
print("finished!")

# In[ ]:


figure.savefig("hw1_gaussian_discriminant.png")

