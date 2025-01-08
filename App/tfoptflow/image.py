

from skimage.io import imread
import numpy as np


image_path1 = f'./samples/mpisintel_test_clean_ambush_1_frame_0001.png'
image_path2 = f'./samples/mpisintel_test_clean_ambush_1_frame_0002.png'
image1, image2 = imread(image_path1), imread(image_path2)

aa1 = image1.shape
aa11 = isinstance(image1, np.ndarray)
aa2 = image2.shape
aa22 = isinstance(image2, np.ndarray)