#!/usr/bin/env python
# coding: utf-8

# # Text Classification
# *Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. Please check the pdf file for more details.*
# 
# In this exercise you will:
#     
# - implement a of spam classifier with **Naive Bayes method** for real world email messages
# - learn the **training and testing phase** for Naive Bayes classifier  
# - get an idea of the **precision-recall** tradeoff

# In[1]:


# some basic imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse



# In[21]:


# ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = np.loadtxt('ham_train.csv', delimiter=',')
# print(type(ham_train), len(ham_train), ham_train.shape)
# spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = np.loadtxt('spam_train.csv', delimiter=',')
# N is the size of vocabulary.
N = ham_train.shape[0]
# There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034
num_spam_train = 3372
# Do smoothing
x = np.vstack([ham_train, spam_train]) + 1

# ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
i,j,ham_test = np.loadtxt('ham_test.txt').T
# change interpreter to python3
i = i.astype(int)
j = j.astype(int)
ham_test = ham_test.astype(int)
ham_test_tight = scipy.sparse.coo_matrix((ham_test, (i - 1, j - 1)))
ham_test = scipy.sparse.csr_matrix((ham_test_tight.shape[0], ham_train.shape[0]))
ham_test[:, 0:ham_test_tight.shape[1]] = ham_test_tight
# spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
i,j,spam_test = np.loadtxt('spam_test.txt').T
# change interpreter to python3
i = i.astype(int)
j = j.astype(int)
spam_test = spam_test.astype(int)
spam_test_tight = scipy.sparse.csr_matrix((spam_test, (i - 1, j - 1)))
spam_test = scipy.sparse.csr_matrix((spam_test_tight.shape[0], spam_train.shape[0]))
spam_test[:, 0:spam_test_tight.shape[1]] = spam_test_tight
# print(spam_test)


# In[3]:


import re
def get_words_dict(path):
    '''
    根据all_word_map.txt文件返回字典
    :param path: 文件路径
    :return: word_dict
    '''
    file = open(path)
    word_dict = {}
    for s_line in file.readlines():
        spliter = re.compile("\\W+")
        words = spliter.split(s_line)
        word_dict[words[0]] = int(words[1])
    file.close()
    return word_dict


# ## Now let's implement a ham/spam email classifier. Please refer to the PDF file for details

# In[4]:


from likelihood import likelihood
# TODO
# Implement a ham/spam email classifier, and calculate the accuracy of your classifier

# begin answer
# a find the top 10 words
p_cx_ham_train = np.zeros([1, N], np.float32)
p_cx_spam_train = np.zeros([1, N], np.float32)

p_cx_ham_train = (ham_train + 1) / (np.sum(ham_train + 1))
p_cx_spam_train = (spam_train + 1) / (np.sum(spam_train + 1))

ratio = []
for i in range(0, N):
    ratio.append(p_cx_spam_train[i] / p_cx_ham_train[i])
    
import heapq
max_idx = heapq.nlargest(10, range(len(ratio)), ratio.__getitem__)
max_idx = [i+1 for i in max_idx]
word_dict = get_words_dict('all_word_map.txt')
top_words_dict = {}
for key, value in word_dict.items():
    if value in max_idx:
        top_words_dict[value] = key
top_words_list = []
for index in max_idx:
    top_words_list.append(top_words_dict[index])
print('top 10 words are: ', top_words_list)

# end answer


# In[ ]:


# b the accuracy of  spam filter on the testing set
# 计算垃圾邮件出现的概率
import math
from scipy.sparse import csr_matrix
spam_positive_prob = num_spam_train / (num_spam_train + num_ham_train)

ham_test = ham_test.toarray()
spam_test = spam_test.toarray()

correct_num = 0
true_labels = []
predict_labels = []
for i in range(0, ham_test.shape[0]):
# for vec in ham_test:
    p_spam = np.sum(np.log(p_cx_spam_train) * ham_test[i]) + math.log(spam_positive_prob)
    p_ham = np.sum(np.log(p_cx_ham_train) * ham_test[i]) + math.log(1 - spam_positive_prob)
    if p_spam > p_ham:
        predict_label = 1
    else:
        predict_label = 0
    predict_labels.append(predict_label)
    if predict_label == 0:
        correct_num += 1        
for i in range(0, spam_test.shape[0]):
# for vec in ham_test:
    p_spam = np.sum(np.log(p_cx_spam_train) * spam_test[i]) + math.log(spam_positive_prob)
    p_ham = np.sum(np.log(p_cx_ham_train) * spam_test[i]) + math.log(1 - spam_positive_prob)
    # p_spam = np.sum(np.log(p_cx_spam_train) * vec) + math.log(spam_positive_prob)
    if p_spam > p_ham:
        predict_label = 1
    else:
        predict_label = 0
    predict_labels.append(predict_label)
    if predict_label == 1:
        correct_num += 1

print(correct_num, len(spam_test), len(ham_test))
accuracy = (correct_num) / (len(spam_test) + len(ham_test))

print('Accuracy: ', accuracy)

true_labels.extend([0] * len(ham_test))
true_labels.extend([1] * len(spam_test))
true_labels = np.array(true_labels)
predict_labels = np.array(predict_labels)

tp = np.sum(np.logical_and(true_labels == 1, predict_labels == 1)) 
fp = np.sum(np.logical_and(true_labels == 0, predict_labels == 1)) 
print('tp: ', tp)
print('fp: ', fp)
precision = tp / (tp + fp)
print('Precision: ', precision)

tp = np.sum(np.logical_and(true_labels == 1, predict_labels == 1)) 
fn = np.sum(np.logical_and(true_labels == 1, predict_labels == 0)) 
print('tp: ', tp)
print('fn: ', fn)
recall = tp / (tp + fn)
print('Recall: ', recall)




# 
