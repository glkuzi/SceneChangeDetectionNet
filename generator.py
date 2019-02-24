# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:48:18 2019

@author: User
"""

import numpy as np
#import cv2
from keras.utils import Sequence
from PIL import Image
DataPath = './dataset/'
Hsize = 512
Wsize = 512
Means = [[107.800, 117.692, 119.979], [110.655, 117.107, 119.135]]


def preprocessing(img, mean):
    img = img.resize((Wsize, Hsize), Image.ANTIALIAS)
    if mean[0] == 0:
        img = img.resize((472, 472), Image.ANTIALIAS)
        return np.array(img.convert('RGB'), dtype='float') / 255
    else:
        temp = np.array(img.convert('RGB'), dtype='float')
        l1 = temp.shape[0]
        l2 = temp.shape[1]
        for i in range(l1):
            for j in range(l2):
                temp[i][j] -= mean
        return temp


class DataGenerator(Sequence):
    def __init__(self, patches, batchSize=32, flag=0):
        self.patches = patches
        self.numb = 0
        self.batchSize = batchSize
        self.X = []
        for j in range(self.batchSize):
            temp = []
            for p in self.patches:
                f = open(p)
                lines = f.readlines()
                buf = Image.open(DataPath + lines[self.numb][:-1])
                if self.patches.index(p) == 0:
                    buf = preprocessing(buf, np.array(Means[0]))
                elif self.patches.index(p) == 1:
                    buf = preprocessing(buf, np.array(Means[1]))
                else:
                    buf = preprocessing(buf, [0, 0, 0])
                temp.append(buf)
            self.numb += 1
            self.X.append(temp)

    def __len__(self):
        return self.batchSize

    def __getitem__(self, index):
        x = [np.array([self.X[index][0]]), np.array([self.X[index][1]])]
        #x1 = np.array(self.X[index][0])
        #x2 = np.array(self.X[index][1])
        y = np.array([self.X[index][-1]])
        return x, y

    def on_epoch_end(self):
        self.X = []
        for j in range(self.batchSize):
            temp = []
            for p in self.patches:
                f = open(p)
                lines = f.readlines()
                buf = Image.open(DataPath + lines[self.numb][:-1])
                if self.patches.index(p) == 0:
                    buf = preprocessing(buf, np.array(Means[0]))
                elif self.patches.index(p) == 1:
                    buf = preprocessing(buf, np.array(Means[1]))
                else:
                    buf = preprocessing(buf, [0, 0, 0])
                temp.append(buf)
            self.numb += 1
            self.X.append(temp)


def main():
    '''
    path = './dataset2014/dataset/dynamicBackground/boats/input/in007157.jpg'
    img = Image.open(path)
    print(np.array(img.convert('RGB')), np.array(img.convert('RGB')).shape)
    '''
    '''
    patches_train = ['t0train.txt', 't1train.txt', 'gttrain.txt']
    patches_val = ['t0val.txt', 't1val.txt', 'gtval.txt']
    dg = DataGenerator(patches_train)
    for i in range(10):
        leng = dg.__len__()
        for j in range(leng):
            x, y = dg.__getitem__(j)
            print(x.shape, y.shape)
    print(x, y)
    '''
    '''
    f = open('val.txt')
    i = 1
    t0 = open('t0val.txt', 'w')
    t1 = open('t1val.txt', 'w')
    gt = open('gtval.txt', 'w')
    for line in f:
        buf = line.split()
        t0.write(buf[0] + '\n')
        t1.write(buf[1] + '\n')
        gt.write(buf[2] + '\n')
        if i == 1:
            buf = line.split()
            print(buf)
            print(line)
            i = 0
    '''
    return 0


main()