
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = '0'
from os.path import join
import argparse
from video2Pics import extract_sequences
from autocrop_faces.autocrop import autocrop_faces
from tfoptflow.pwcnet_predict_from_img_pairs2 import pwcnet
from tfrecord.GenerateTFRecord import GenerateTFRecord
import tensorflow as tf
from tensorflow.keras import models, datasets
from tensorflow.keras.layers import Dense,TimeDistributed, Dropout, Embedding, LSTM, Bidirectional, Reshape, Input, Flatten, BatchNormalization, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import applications
import statistics
import numpy as np
import argparse
import sys


class deepfake_console:
    
    
    def get_tfrecords_features(self):
        print("get_tfrecords_features")
        return {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64)
        }
    
    def feature_retrieval(self, serialized_example):
        print("feature_retrieval")
        
        example  = tf.io.parse_single_example(serialized_example, features=self.get_tfrecords_features())
        
        
        image = tf.cast(example ['image'], tf.string)
        height = tf.cast(example ['height'], tf.int64)
        width = tf.cast(example ['width'], tf.int64)
        label = tf.cast(example ['label'], tf.int64)
        
        image_shape = tf.stack([height, width, 3])
        image_raw = tf.io.decode_raw(image, tf.float32)
        image = tf.reshape(image_raw, image_shape)

        image = tf.image.resize(image, [120, 120])
        label = tf.one_hot(label, 1)
        #label.set_shape([None,1])
        
        return image, label
    
    
        
    def prepare_data(self, filename_queue, nb_images):
        
        # This works with arrays as well, num_parallel_reads = 8
        #dataset = tf.data.TFRecordDataset(filenames = filename_queue, num_parallel_reads = 6)
        files = tf.data.Dataset.list_files(filename_queue)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(tf.data.experimental.AUTOTUNE))


        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self.feature_retrieval, num_parallel_calls = tf.data.experimental.AUTOTUNE)

        # This dataset will go on forever

        # Set the number of datapoints you want to load and shuffle 
        dataset = dataset.shuffle(200)

        # Set the batchsize, drop_remainder=True
        batch_size = nb_images
        dataset = dataset.repeat(tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(1) 
        
        return dataset
    
    
    def vgg16(self, dataset):
        
        input_tensor = Input(shape=(120, 120, 3))
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
        # Freeze the layers except the last 4 layers
        for layer in vgg_model.layers[:-4]:
            layer.trainable = False
        
        # Create the model
        model = models.Sequential()
    
         # Add the vgg convolutional base model
        model.add(vgg_model)
        # Classification block
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dense(1, activation='sigmoid', name='predictions'))
        adam = optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        # Loads the weights
        model.load_weights("cnn1.h5")
        
        # evaluate the model
        acc = model.predict(dataset, steps=1)
        print("++++++++++++++++++++++++++++++++++++++++")
        print(statistics.mean(acc.ravel()))
        if statistics.mean(acc.ravel()) <= 0.5:
            print("Original")
        else:
            print("Fake")
        
        
        
    
    
if __name__ == '__main__':
    
    print("Processing images in folder:", input)
    
    p = argparse.ArgumentParser()
    p.add_argument('--video', '-i',
                   default=None)
    
    args = p.parse_args()
    vargs = vars(args)
    
    if args.video == None:
        print('Need a video')
        sys.exit()
    
    video_path = args.video
    
    
    images_dir, nb_images = extract_sequences(video_path)
    
    video_dir = autocrop_faces(images_dir)
    optical_flow = pwcnet(video_dir)
    
    tfrecords_path = GenerateTFRecord().run(optical_flow, video_dir)
    tfrecords_path = tf.io.gfile.glob(tfrecords_path)
    
    df = deepfake_console()
    dataset = df.prepare_data(tfrecords_path, nb_images)
    df.vgg16(dataset)
        
        
            