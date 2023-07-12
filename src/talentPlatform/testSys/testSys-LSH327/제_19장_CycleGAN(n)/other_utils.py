# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:39:53 2020

@author: ysp
"""
import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:],[0.299,0.587,0.114])

def display_images(imgs,title='',show=False):
    rows=imgs.shape[1]
    cols=imgs.shape[2]
    channels=imgs.shape[3]
    side=int(np.sqrt(imgs.shape[0]))
    if channels==1:
        imgs=imgs.reshape((side,side,rows,cols))
    else:
        imgs=imgs.reshape((side,side,rows,cols,channels))
    imgs=np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title(title)
    if channels==1:
        plt.imshow(imgs,cmap='gray')
    else:
        plt.imshow(imgs)
    if show:
        plt.show()
    plt.close('all')
    
def load_data(data,titles,todisplay=25):
    A_data, B_data, test_A_data, test_B_data=data
    test_A_title, test_B_title=titles
    
    imgs_B=test_B_data[:todisplay]
    display_images(imgs_B, title=test_B_title)
    imgs_A=test_A_data[:todisplay]
    display_images(imgs_A,title=test_A_title)
    
    B_data=B_data.astype('float32')/255
    test_B_data=test_B_data.astype('float32')/255
    A_data=A_data.astype('float32')/255
    test_A_data=test_A_data.astype('float32')/255
    
    data=(A_data, B_data, test_A_data, test_B_data)
    height_A=A_data.shape[1]
    width_A=A_data.shape[2]
    channel_A=A_data.shape[3]
    A_shape=(height_A,width_A,channel_A)
    
    height_B=B_data.shape[1]
    width_B=B_data.shape[2]
    channel_B=B_data.shape[3]
    B_shape=(height_B,width_B,channel_B)
    
    shapes=(A_shape,B_shape)
    return data,shapes

def test_generator(generators, test_data, step, titles, todisplay=4,show=False):
    g_BA, g_AB=generators # monet_photo이외에는 todisplay=25로 지정
    test_A_data, test_B_data=test_data 
    t1,t2,t3,t4=titles
    title_BA=t1
    title_AB=t2
    title_reco_BA=t3
    title_reco_AB=t4
    
    pred_AB=g_AB.predict(test_A_data)
    pred_AB=np.clip(pred_AB,0,1)
    pred_BA=g_BA.predict(test_B_data)
    pred_BA=np.clip(pred_BA,0,1)
    reco_AB=g_AB.predict(pred_BA)
    reco_AB=np.clip(reco_AB,0,1)
    reco_BA=g_BA.predict(pred_AB)
    reco_BA=np.clip(reco_BA,0,1)
        
    imgs=pred_AB[:todisplay]
    step="Step:{:,}".format(step)
    title=title_AB+step
    display_images(imgs,title=title,show=show)
    
    imgs=pred_BA[:todisplay]
    title=title_BA
    display_images(imgs,title=title,show=show)
    
    imgs=reco_BA[:todisplay]
    title=title_reco_BA
    display_images(imgs,title=title,show=show)
    
    imgs=reco_AB[:todisplay]
    title=title_reco_AB
    display_images(imgs,title=title,show=show)

from scipy import io    
def loadmat(data_mat):
    data=data_mat['X']
    data=np.transpose(data,[3,0,1,2])
    return data
    
    
        
        