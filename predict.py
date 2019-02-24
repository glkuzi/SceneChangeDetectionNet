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
    model_path = 'testn.01-nan.hdf5'
    path1 = 'C:\Download\SevStal\TESTSET\IMG_20190223_140201.jpg'
    path2 = 'C:\Download\SevStal\TESTSET\IMG_20190223_140210.jpg'
    buf1 = preprocessing(Image.open(path1), np.array(Means[0]))
    buf2 = preprocessing(Image.open(path2), np.array(Means[1]))
    X = [np.array([buf1]), np.array([buf2])]
    model = load_model(model_path)
    Y = model.predict(X)
    print(Y)
    img = Image.fromarray(Y)
    Image.SAVE('test.png', img)
    return 0


main()
