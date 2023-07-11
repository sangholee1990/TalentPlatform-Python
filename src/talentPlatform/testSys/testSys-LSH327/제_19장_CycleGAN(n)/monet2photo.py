# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:45:17 2020

@author: ysp
"""

import imageio
import glob
#import scipy
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
def load_batch(img_res=(256,256), is_train=True):
    if is_train:
        data_type='train'
    else:
        data_type='test'
    path_A=glob.glob('C:/Users/ysp/Desktop/Deep Learning/GDL_code-master/monet2photo/%sA/*' %data_type)
    path_B=glob.glob('C:/Users/ysp/Desktop/Deep Learning/GDL_code-master/monet2photo/%sB/*' %data_type)
    
    total_sample=int(min(len(path_A),len(path_B)))
        
    path_A=np.random.choice(path_A,total_sample,replace=False)
    path_B=np.random.choice(path_B,total_sample,replace=False)
    
    imgs_A,imgs_B=[],[]
    
    for i in range(total_sample):
        batch_A=path_A[i]
        batch_B=path_B[i]
                
        img_A=imageio.imread(batch_A).astype(np.float)
        img_B=imageio.imread(batch_B).astype(np.float)
        img_A=resize(img_A, img_res)
        img_B=resize(img_B, img_res)
            
        imgs_A.append(img_A)
        imgs_B.append(img_B)
    imgs_A=np.array(imgs_A)/127.5-1
    imgs_B=np.array(imgs_B)/127.5-1
    return imgs_A, imgs_B


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
    
def load_data(data,titles,todisplay=4):
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
    g_BA, g_AB=generators
    test_A_data, test_B_data=test_data
    t1,t2,t3,t4=titles
    title_BA=t1
    title_AB=t2
    title_reco_BA=t3
    title_reco_AB=t4

    imgs_A=test_A_data[:todisplay]
    imgs_A=(imgs_A+1)*127.5/255.
    title='orininal_Monet'
    display_images(imgs_A,title=title,show=show)
    imgs_B=test_B_data[:todisplay]
    imgs_B=(imgs_B+1)*127.5/255.
    title='orininal_photo'
    display_images(imgs_B,title=title,show=show)
    
    
    
    pred_AB=g_AB.predict(test_A_data)
    pred_AB=(pred_AB+1)*127.5/255.
    pred_AB=np.clip(pred_AB,0,1)
    pred_BA=g_BA.predict(test_B_data)
    pred_BA=(pred_BA+1)*127.5/255.
    pred_BA=np.clip(pred_BA,0,1)
    reco_AB=g_AB.predict(pred_BA)
    reco_AB=(reco_AB+1)*127.5/255.
    reco_AB=np.clip(reco_AB,0,1)
    reco_BA=g_BA.predict(pred_AB)
    reco_BA=(reco_BA+1)*127.5/255.
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

