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



class readTFRecord:
    
    
    def get_tfrecords_features(self):
        print("get_tfrecords_features")
        return {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64)
        }
    
    def load_tfrecords(self, tfrecords_filepath):
        items = []
        labels = []
        print("Loading %s" % tfrecords_filepath)
        with tf.Session() as sess:
            for serialized_example in tf.python_io.tf_record_iterator(tfrecords_filepath):
                data, label = self.feature_retrieval(serialized_example)
                items.append(data)
                labels.append(label)
        print("Finished Loading %s" % tfrecords_filepath)
        return (tf.stack(items), tf.stack(labels))
    
    def feature_retrieval(self, serialized_example):
        print("feature_retrieval")
        
        example  = tf.parse_single_example(serialized_example, features=self.get_tfrecords_features())
        
        
        image = tf.cast(example ['image'], tf.string)
        height = tf.cast(example ['height'], tf.int64)
        width = tf.cast(example ['width'], tf.int64)
        label = tf.cast(example ['label'], tf.int64)
        
        image_shape = tf.stack([height, width, 3])
        image_raw = tf.decode_raw(image, tf.float32)
        image = tf.reshape(image_raw, image_shape)

        image = tf.image.resize_images(image, [120, 120])
        label = tf.one_hot(label, 1)
        #label.set_shape([None,1])
        
        return image, label
    
    
        
    def prepare_data(self, filename_queue):
        print("prepare_data")
        
        # This works with arrays as well, num_parallel_reads = 8
        #dataset = tf.data.TFRecordDataset(filenames = filename_queue, num_parallel_reads = 6)
        files = tf.data.Dataset.list_files(filename_queue)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100), cycle_length=2, block_length=128)


        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self.feature_retrieval, num_parallel_calls = 2)

        # This dataset will go on forever

        # Set the number of datapoints you want to load and shuffle 
        #dataset = dataset.shuffle(200)

        # Set the batchsize, drop_remainder=True
        batch_size = 256
        dataset = dataset.shuffle(1000 + 3 * batch_size )
        dataset = dataset.repeat(100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1) 
        """ 
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        
        print(label_batch)
        with tf.Session() as sess:
            image_batch, label_batch = sess.run([image_batch, label_batch])
            print(tf.shape(image_batch))
          
        sess = tf.Session()
        image_batch, label_batch = sess.run([image_batch, label_batch])
        print(tf.shape(image_batch))
        

        dataset = tf.data.Dataset.list_files(filename_queue)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16)
        dataset = dataset.map(self.feature_retrieval, num_parallel_calls=AUTO)

        dataset = dataset.cache() # This dataset fits in RAM
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(2000, drop_remainder=True) # drop_remainder will be needed on TPU"""
       
        return dataset