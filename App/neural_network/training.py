from __future__ import absolute_import, division, print_function
"""import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'"""
import sys

import tensorflow as tf
import glob, os
from tensorflow.python.lib.io import file_io
from tensorflow.keras import applications
from tensorflow.keras import models, datasets
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Reshape, Input, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import Callback
import numpy as np
import time
from random import shuffle
np.set_printoptions(threshold=sys.maxsize)
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from readTFRecord import readTFRecord
from vgg16 import vgg16


class DeepFakeTraining:

    def prepare_training_data(self, opticalFlowOriginalPath, opticalFlowFakePath):
        original = tf.gfile.Glob(opticalFlowOriginalPath)
        fake = tf.gfile.Glob(opticalFlowFakePath)
        
        original_size = len(original)
        original_train = original[0 : int(0.8*original_size)]
        original_val = original[int(0.8*original_size) : int(0.9*original_size)]
        original_test = original[int(0.9*original_size) : -1]
        
        
        fake_size = len(fake)
        fake_train = fake[0 : int(0.8*fake_size)]
        fake_val = fake[int(0.8*fake_size) : int(0.9*fake_size)]
        fake_test = fake[int(0.9*fake_size) : -1]
        
        
        min_train = min(len(original_train), len(fake_train))
        data_train = [None]*(min_train*2)
        data_train[::2] = original_train[:min_train]
        data_train[1::2] = fake_train[:min_train]
        data_train.extend(original_train[min_train:])
        data_train.extend(fake_train[min_train:])
        
        min_val = min(len(original_val), len(fake_val))
        data_val = [None]*(min_val*2)
        data_val[::2] = original_val[:min_val]
        data_val[1::2] = fake_val[:min_val]
        data_val.extend(original_val[min_val:])
        data_val.extend(fake_val[min_val:])
        
        min_test = min(len(original_test), len(fake_test))
        data_test = [None]*(min_test*2)
        data_test[::2] = original_test[:min_test]
        data_test[1::2] = fake_test[:min_test]
        data_test.extend(original_test[min_test:])
        data_test.extend(fake_test[min_test:])
        
        return data_train, data_val, data_test  
        

if __name__ == '__main__':
    
    original = "D:/tfo/*.tfrecords"
    fake = "D:/tff/*.tfrecords"
    
    training = DeepFakeTraining()
    data_train, data_val, data_test = training.prepare_training_data(original, fake)
    print(len(data_train))
    print(len(data_val))   
    print(len(data_test))   
    
    r = readTFRecord()
    training_dataset = r.prepare_data(data_train)
    validation_dataset = r.prepare_data(data_val)
    test_dataset = r.prepare_data(data_test)
    
    m = applications.VGG16(weights='imagenet')
    #m.summary()
    
    vgg16().model(training_dataset, validation_dataset, test_dataset)
    