# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:09:59 2020

@author: ysp
"""

from tensorflow.keras.datasets import cifar10
#from skimage import color
import numpy as np
import other_utils
def load_cifar10():
    (B_data,_),(test_B_data,_)=cifar10.load_data()
    A_data=other_utils.rgb2gray(B_data)
    test_A_data=other_utils.rgb2gray(test_B_data)
    A_data=A_data[:,:,:,np.newaxis]
    test_A_data=test_A_data[:,:,:,np.newaxis]
    data=(A_data,B_data,test_A_data,test_B_data)
    titles=('CIFAR10 test_A_data images', 'CIFAR10 test_A_data images')
    return other_utils.load_data(data,titles)