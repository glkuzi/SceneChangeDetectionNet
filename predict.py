# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 05:26:06 2019

@author: User
"""

import numpy as np
from generator import *
from PIL import Image
from keras.models import load_model
#from tensorflow import norm


def main():
    model_path = 'testn.01-0.05.hdf5'
    path1 = 'C:\Download\SevStal\TESTSET\IMG_20190223_140201.jpg'
    path2 = 'C:\Download\SevStal\TESTSET\IMG_20190223_140210.jpg'
    tpath = '‪C:\Download\SevStal\dataset\PTZ\twoPositionPTZCam\gt_binary\gt001386.png'
    buf1 = preprocessing(Image.open(path1), np.array(Means[0]))
    buf2 = preprocessing(Image.open(path2), np.array(Means[1]))
    targ = preprocessing(Image.open(path2), [0, 0, 0])
    print(targ, np.max(targ))
    X = [np.array([buf1]), np.array([buf2])]
    
    model = load_model(model_path)
    Y = model.predict(X)
    print(Y, np.max(Y / np.max(Y)))
    Y = Y /np.max(Y)
    Y = np.array(Y * 255, dtype='uint8')[0]
    print(Y, Y.shape)
    img = Image.fromarray(Y)
    print(img)
    img.save('test.png')
    
    return 0


main()
