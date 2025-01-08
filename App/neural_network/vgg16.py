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


class vgg16:
    
    
    
    def model(self, training_dataset, validation_dataset, test_dataset):
        input_tensor = Input(shape=(120, 120, 3))
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
        # Freeze the layers except the last 4 layers
        for layer in vgg_model.layers[:-4]:
            layer.trainable = False
    
        # Check the trainable status of the individual layers
        for layer in vgg_model.layers:
            print(layer, layer.trainable)
        
        # Create the model
        model = models.Sequential()
    
        # Add the vgg convolutional base model
        model.add(vgg_model)
        
        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(1, activation='sigmoid', name='predictions'))
        
        #lr_sched = r.step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=976)
        
        adam = optimizers.Adam(lr=0.00000000001)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        
    
        # Show a summary of the model. Check the number of trainable parameters
        model.summary()
        model_name = "deepfake_model-{}".format(int(time.time()))
        checkpoint = ModelCheckpoint("best_deepfake_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
        
        history = model.fit(training_dataset, validation_data=validation_dataset,validation_steps=140,
                            epochs=6,
                            steps_per_epoch=976,
                            callbacks=[checkpoint])
        
        # The returned "history" object holds a record
        # of the loss values and metric values during training
        print('\nhistory dict:', history.history)
        
        # Save the model
        model.save(model_name)
        
        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        results = model.evaluate(test_dataset, steps=140)
        print('test loss, test acc:', results)
        
        