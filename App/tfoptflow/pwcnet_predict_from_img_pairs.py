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
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows, archive_img_pairs_w_flows
import numpy as np
from PIL import Image

# TODO: Set device to use for inference
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:CPU:0']  
controller = '/device:CPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = './models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

# Build a list of image pairs to process
img_pairs = []
for pair in range(0, 4):
    print(pair)
    image_path1 = f'./samples/00{pair:02d}.png'
    image_path2 = f'./samples/00{pair+1:02d}.png'
    """if (pair < 100):
        image_path1 = f'./samples/00{pair:02d}.png'
        if (pair < 99):
            image_path2 = f'./samples/00{pair+1:02d}.png'
        else:
            image_path2 = f'./samples/0{pair+1:02d}.png'
    else:
        image_path1 = f'./samples/0{pair:02d}.png'
        image_path2 = f'./samples/0{pair+1:02d}.png'"""
    image1, image2 = imread(image_path1), imread(image_path2)
    img_pairs.append((image1, image2))

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

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 269, 269, 2)

# Instantiate the model in inference mode and display the model configuration
nn = ModelPWCNet(mode='test', options=nn_opts)
nn.print_config()

# Generate the predictions and display them
pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)




display_img_pairs_w_flows(img_pairs, pred_labels)
#♥archive_img_pairs_w_flows(img_pairs, 'C:/Users/Dell/Desktop/pwc/tfoptflow/samples/res/')