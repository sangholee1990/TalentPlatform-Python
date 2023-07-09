# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:16:02 2020

@author: ysp
"""
from tensorflow.keras.datasets import mnist
import numpy as np
from scipy import io
import other_utils

def load_mnist_svhn():
    (A_data,_),(test_A_data,_)=mnist.load_data()
    A_data=np.pad(A_data,((0,0),(2,2),(2,2)),'constant',constant_values=0)
    test_A_data=np.pad(test_A_data,((0,0),(2,2),(2,2)),'constant',constant_values=0)
    A_data=A_data[:,:,:,np.newaxis]
    test_A_data=test_A_data[:,:,:,np.newaxis]
    B_data_mat=io.loadmat("C:/Users/ysp/Desktop/Deep Learning/train_32x32.mat")
    test_B_mat=io.loadmat("C:/Users/ysp/Desktop/Deep Learning/test_32x32.mat")
    B_data=other_utils.loadmat(B_data_mat)
    test_B_data=other_utils.loadmat(test_B_mat)
    
    data=(A_data,B_data,test_A_data,test_B_data)
    titles=('MNIST test_A_data images','SVHN test_B_data images')
    return other_utils.load_data(data,titles)