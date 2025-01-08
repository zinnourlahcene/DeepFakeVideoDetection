"""
pwcnet_predict_from_img_pairs.py

Run inference on a list of images pairs.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from copy import deepcopy
from skimage.io import imread
from .model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from optflow import flow_write_as_png
from os.path import join

def pwcnet(video_dir):
    # TODO: Set device to use for inference
    # Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
    gpu_devices = ['/device:CPU:0']  
    controller = '/device:CPU:0'
    
    # TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
    ckpt_path = r'D:/Deepfake/App/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
    
    # Configure the model for inference, starting with the default options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller
    
    # We're running the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2
    #-----------------------------------------------------------------------------------
    
    faces_dir = join(video_dir, "faces")
    opticalFlow_dir = join(video_dir, "opticalFlow")
    os.makedirs(opticalFlow_dir)
    
    # Build a list of image pairs to process
    faces_pairs = []
    path, dirs2, files = next(os.walk(faces_dir))
    h, w, c = 0, 0, 0
    for pair in range(0, len(files)):
        if(pair < len(files)-1):
            image_path1 = path+"/"+files[pair]
            image_path2 = path+"/"+files[pair+1]
        
            image1, image2 = imread(image_path1), imread(image_path2)
            faces_pairs.append((image1, image2))
            
            h, w, c = image1.shape

    # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
    # of 64. Hence, we need to crop the predicted flows to their original size
    nn_opts['adapt_info'] = (1, h, h, 2)
    
    # Instantiate the model in inference mode and display the model configuration
    nn = ModelPWCNet(mode='test', options=nn_opts)
    nn.print_config()
    
    # Generate the predictions and display them
    pred_labels = nn.predict_from_img_pairs(faces_pairs, batch_size=1, verbose=False)
    i = 0
    for img in pred_labels:
        f = join(opticalFlow_dir, files[i])
        flow_write_as_png(img, f)
        i += 1

    return opticalFlow_dir

    