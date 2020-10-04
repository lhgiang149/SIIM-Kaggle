import numpy as np
import pandas as pd 
import cv2
import os
import tqdm

'''
    Normalize in albumentation from ImageNet, which is (follow me) not suitable for this 
competition. So this function use for take new mean and variance for normalize that dataset.
'''

class Params:
    def __init__(self):
        # self.dataroot = 'E:/SIIM/train/'
        self.dataroot = 'C:/Users/ADMINS/Desktop/SIIM-ISIC-Melanoma-Classification/data/train/'
        
params = Params()

def _normalize(image):
    if len(image.shape) != 3:
        raise Exception("Need color image")
    max_value = np.iinfo(image.dtype).max
    # print(int(max_value))
    return image/max_value

def _mean(image):
    return np.mean(image, axis = (0,1))

def _var(image):
    return np.var(image, axis = (0,1))

def _process_image(pth):
    image = cv2.imread(params.dataroot + pth)
    image = _normalize(image)
    return _mean(image), _var(image)

# def _main_process(pth):
#     num_sample = len(os.listdir(dataroot))
#     sum_m = sum_std = np.zeros((1,3)) 
#     image = cv2.imread(dataroot + pth) 
#     m, std = _process(image)
#     params.global_mean += m 
#     params.global_std += std
#     return m,    

if __name__ == "__main__":
    from multiprocessing import Pool

    dataroot = params.dataroot
    num_sample = len(os.listdir(dataroot))

    pool = Pool(processes=8)
    package = pool.map(_process_image, os.listdir(dataroot))
    package = np.array(package)
    mean = np.sum(package[:,0,:], axis = 0)/num_sample
    var = np.sum(package[:,1,:], axis = 0)/num_sample
    std = np.sqrt(var)
    print('mean: ', mean)
    print('std: ', std)
    with open('result.txt', 'w') as f:
        for i in mean:
            f.write(str(i) + '\n')
        for i in std:
            f.write(str(i) + '\n')
        # print(a/num_sample, b/num_sample)
    
    # print(mean, std)
