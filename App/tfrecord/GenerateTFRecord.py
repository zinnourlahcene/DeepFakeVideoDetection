

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from copy import deepcopy
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from os.path import join

class GenerateTFRecord:
    
        
    def _int64_feature(self, value):
        return tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value])
             )
    def _floats_feature(self, value):
        return tf.train.Feature(
                   float_list=tf.train.FloatList(value=value)
               )
    
    def _bytes_feature(self, value):
        return tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[value])
             )
      
      
    def load_image(self, path):
        img = cv2.imread(path)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img
    
    def get_feature(self, image, label, height, width):
        return {
                'image': self._bytes_feature(tf.compat.as_bytes(image.tostring())),
                'label': self._int64_feature(label),
                'height': self._int64_feature(height),
                'width': self._int64_feature(width)
                }
    
    def createTfRecord(self, images_path, label, height, width, output_file_path):
        with tf.io.TFRecordWriter(output_file_path) as writer:
            for index in range(len(images_path)):
                img = self.load_image(images_path[index])
                example = tf.train.Example(
                  features=tf.train.Features(
                      feature = self.get_feature(img, label, height, width)
                ))
                writer.write(example.SerializeToString())
                print('\r{:.1%}'.format((index+1)/len(images_path)), end='')
                
                
    def run(self, input_path, output_path):
        output_path = join(output_path, "tfrecords")
        os.makedirs(output_path)
        imgs_path = []
        
        #To extract height and width
        path, dirs2, files = next(os.walk(input_path))
        image_path = input_path+"/"+files[0]
        image = imread(image_path)
        h, w, c = image.shape
        
        for images in os.listdir(input_path):
            img_path = join(input_path +'/'+ images)
            imgs_path.append(img_path)
        tf = join(output_path, "tf.tfrecords")
        self.createTfRecord(imgs_path, 1, h, w, tf)
        
        return tf
    
    


