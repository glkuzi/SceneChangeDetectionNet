# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:47:44 2019

@author: User
"""

import tensorflow as tf
import numpy as np
import keras.backend as K
from tensorflow.math import reduce_max
from generator import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda, UpSampling2D, BatchNormalization
from keras.models import Sequential, Model
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from scipy.misc import imresize
#from keras.utils import normalize
from tensorflow.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd


def toTarget(tensor):
    buf1 = reduce_max(K.flatten(tensor))
    print(buf1)
    buf2 = tensor
    return imresize(buf2 / buf1, 8.0)
    #return imresize(np.array(tensor, dtype='float') / float(K.max(K.flatten(tensor))), 8.0)


def normal(tensors):
    tensor = K.sqrt(K.square(tensors[0]) - K.square(tensors[1]))
    tensor = tf.div(tf.subtract(tensor, tf.reduce_min(tensor)), tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
    return tensor


def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(16, (10, 10), activation='sigmoid', input_shape=input_shape, #64
                     kernel_regularizer=regularizers.l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(8, (7, 7), #128
                     kernel_regularizer=regularizers.l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(4, (4, 4), #128
                     kernel_regularizer=regularizers.l2(2e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(3, (2, 2),
                     kernel_regularizer=regularizers.l2(2e-4)))
    #model.add(Conv2D(3, (4, 4), activation='relu',
    #                 kernel_regularizer=regularizers.l2(2e-4)))
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    # Add a customized layer to compute the absolute difference between the encodings
    #L2_layer = Lambda(lambda tensors:(normal(tensors)))#K.sqrt(K.square(tensors[0]) - K.square(tensors[1])) / norm(K.sqrt(K.square(tensors[0]) - K.square(tensors[1])), ord=1))
    #L2_layer = Lambda(normal, arguments=tensors)
    #L2_layer = Lambda(lambda tensors:K.normalize_batch_in_training( K.sqrt(K.square(tensors[0]) - K.square(tensors[1])), 
    #                                                               K.ones_like(K.sqrt(K.square(tensors[0]) - K.square(tensors[1]))), 
    #                                                               K.zeros_like(K.sqrt(K.square(tensors[0]) - K.square(tensors[1]))) ))
    L2_layer = Lambda(lambda tensors:(K.sqrt(K.square(tensors[0]) - K.square(tensors[1]))))
    L2_distance = L2_layer([encoded_l, encoded_r])
    UpLayer = UpSampling2D(size=(8, 8))(L2_distance)
    NormLayer = BatchNormalization()(UpLayer)
    #UpLayer = Lambda(lambda tensor:toTarget(tensor))(L2_distance)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    #prediction = Dense(1,activation='sigmoid')(L2_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=UpLayer)
    # return the model
    return siamese_net


def main():
    inp_shape = (512, 512, 3)
    patches_train = ['t0train.txt', 't1train.txt', 'gttrain.txt']
    patches_val = ['t0val.txt', 't1val.txt', 'gtval.txt']
    dg = DataGenerator(patches_train)
    valid = DataGenerator(patches_val)
    model = get_siamese_model(inp_shape)
    check = ModelCheckpoint('testn.{epoch:02d}-{val_loss:.2f}.hdf5', period=1)
    model.compile(loss='mae', optimizer=optimizers.Adam(lr=0.0001, decay=0.0001), metrics=['accuracy'])
    history = model.fit_generator(generator=dg, steps_per_epoch=32, epochs=50, verbose=1, callbacks=[check],
                        validation_data=valid)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('test.csv', index=False, index_label=False)
    
    return 0


main()
