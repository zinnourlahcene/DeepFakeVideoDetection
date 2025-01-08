# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import argparse
import cv2
import io
import numpy as np
import os
import shutil
import sys
import string
import random
from PIL import Image



FIXEXP = True  # Flag to fix underexposition
MINFACE = 8  # Minimum face size ratio; too low and we get false positives
INCREMENT = 0.06
GAMMA_THRES = 0.001
GAMMA = 0.90
FACE_RATIO = 6  # Face / padding ratio
QUESTION_OVERWRITE = "Overwrite image files?"

# File types supported by OpenCV
CV2_FILETYPES = [
    ".bmp",
    ".dib",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".ppm",
    ".ras",
    ".sr",
    ".tif",
    ".tiff",
    ".webp",
]

# File types supported by Pillow
PILLOW_FILETYPES = [
    ".eps",
    ".gif",
    ".icns",
    ".ico",
    ".im",
    ".msp",
    ".pcx",
    ".sgi",
    ".spi",
    ".xbm",
]

CASCFILE = "haarcascade_frontalface_default.xml"


COMBINED_FILETYPES = CV2_FILETYPES + PILLOW_FILETYPES
INPUT_FILETYPES = COMBINED_FILETYPES + [s.upper() for s in COMBINED_FILETYPES]

# Load XML Resource
# Load XML Resource
d = os.path.dirname("D:/Deepfake/Data preprocessing/autocrop/")
cascPath = os.path.join(d, CASCFILE)
print(cascPath)

# Load custom exception to catch a certain failure type
class ImageReadError(BaseException):
    pass


# Define simple gamma correction fn
def gamma(img, correction):
    img = cv2.pow(img / 255.0, correction)
    return np.uint8(img * 255)

h1 = 0
h2 = 0
v1 = 0
v2 = 0
def crop_positions(
    i,
    imgh,
    imgw,
    x,
    y,
    w,
    h,
    fheight,
    fwidth,
    facePercent,
    padUp,
    padDown,
    padLeft,
    padRight,
):
    global h1
    global h2
    global v1
    global v2
    # Check padding values
    padUp = 50 if (padUp is False or padUp < 0) else padUp
    padDown = 50 if (padDown is False or padDown < 0) else padDown
    padLeft = 50 if (padLeft is False or padLeft < 0) else padLeft
    padRight = 50 if (padRight is False or padRight < 0) else padRight

    # enforce face percent
    facePercent = 100 if facePercent > 100 else facePercent
    facePercent = 50 if facePercent <= 0 else facePercent

    # Adjust output height based on Face percent
    height_crop = h * 100.0 / facePercent

    aspect_ratio = float(fwidth) / float(fheight)
    # Calculate width based on aspect ratio
    width_crop = aspect_ratio * float(height_crop)

    # Calculate padding by centering face
    xpad = (width_crop - w) / 2
    ypad = (height_crop - h) / 2

    # Calc. positions of crop
    if(i):
        h1 = float(x - (xpad * padLeft / (padLeft + padRight)))
        h2 = float(x + w + (xpad * padRight / (padLeft + padRight)))
        v1 = float(y - (ypad * padUp / (padUp + padDown)))
        v2 = float(y + h + (ypad * padDown / (padUp + padDown)))
    """  
    print("h1")
    print(h1)
    print("h2")
    print(h2)
    print("v1")
    print(v1)
    print("v2")
    print(v2)"""

    return adjust_crop_to_boundaries(
        imgh,
        imgw,
        h1,
        h2,
        v1,
        v2,
        aspect_ratio,
        padLeft / (padLeft + padRight),
        padLeft / (padLeft + padRight),
        padUp / (padUp + padDown),
        padDown / (padUp + padDown)
    )


# Move crop inside photo boundaries
def adjust_crop_to_boundaries(
        imgh,
        imgw,
        h1,
        h2,
        v1,
        v2,
        aspect_ratio,
        leftPadRatio,
        rightPadRatio,
        topPadRatio,
        bottomPadRatio
):

    # Calculate largest horizontal/vertical out of bound value
    # with padding ratios in mind
    delta_h = 0.0
    if h1 < 0:
        delta_h = abs(h1) / leftPadRatio

    if h2 > imgw:
        delta_h = max(delta_h, (h2 - imgw) / rightPadRatio)

    delta_v = 0.0 if delta_h <= 0.0 else delta_h / aspect_ratio

    if v1 < 0:
        delta_v = max(delta_v, abs(v1) / topPadRatio)

    if v2 > imgh:
        delta_v = max(delta_v, (v2 - imgh) / bottomPadRatio)

    delta_h = max(delta_h, delta_v * aspect_ratio)

    # Adjust crop values accordingly
    h1 += delta_h * leftPadRatio
    h2 -= delta_h * rightPadRatio
    v1 += delta_v * topPadRatio
    v2 -= delta_v * bottomPadRatio

    return [int(v1), int(v2), int(h1), int(h2)]


def crop(
    image,
    i,
    fheight=500,
    fwidth=500,
    facePercent=50,
    padUp=False,
    padDown=False,
    padLeft=False,
    padRight=False,
):
    """Given a ndarray image with a face, returns cropped array.

    Arguments:
        - image, the numpy array of the image to be processed.
        - fwidth, the final width (px) of the cropped img. Default: 500
        - fheight, the final height (px) of the cropped img. Default: 500
        - facePercent, percentage of face height to image height. Default: 50
        - padUp, Padding from top
        - padDown, Padding to bottom
        - padLeft, Padding from left
        - padRight, Padding to right
    Returns:
        - image, a cropped numpy array

    ndarray, int, int -> ndarray
    """
    # Some grayscale color profiles can throw errors, catch them
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = image

    # Scale the image
    try:
        height, width = image.shape[:2]
    except AttributeError:
        raise ImageReadError
    minface = int(np.sqrt(height ** 2 + width ** 2) / MINFACE)

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # ====== Detect faces in the image ======
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(minface, minface),
        flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
    )

    # Handle no faces
    if len(faces) == 0:
        return None

    # Make padding from biggest face found
    x, y, w, h = faces[-1]

    pos = crop_positions(
        i,
        height,
        width,
        x,
        y,
        w,
        h,
        fheight,
        fwidth,
        facePercent,
        padUp,
        padDown,
        padLeft,
        padRight,
    )
    # Actual cropping
    image = image[pos[0]: pos[1], pos[2]: pos[3]]

    # Resize
    #image = cv2.resize(image, (fwidth, fheight), interpolation=cv2.INTER_AREA)

    # ====== Dealing with underexposition ======
    if FIXEXP:
        # Check if under-exposed
        uexp = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if sum(uexp[-26:]) < GAMMA_THRES * sum(uexp):
            image = gamma(image, GAMMA)
    return image


def open_file(input_filename):
    """Given a filename, returns a numpy array"""
    extension = os.path.splitext(input_filename)[1].lower()

    if extension in CV2_FILETYPES:
        # Try with cv2
        return cv2.imread(input_filename)
    if extension in PILLOW_FILETYPES:
        # Try with PIL
        with Image.open(input_filename) as img_orig:
            return np.asarray(img_orig)
    return None


def output(input_filename, output_filename, image):
    """Move the input file to the output location and write over it with the
    cropped image data."""
    if input_filename != output_filename:
        # Move the file to the output directory
        shutil.move(input_filename, output_filename)
    # Encode the image as an in-memory PNG
    img_png = cv2.imencode(".png", image)[1].tostring()
    # Read the PNG data
    img_new = Image.open(io.BytesIO(img_png))
    # Write the new image (converting the format to match the output
    # filename if necessary)
    img_new.save(output_filename)


def reject(input_filename, reject_filename):
    """Move the input file to the reject location."""
    if input_filename != reject_filename:
        # Move the file to the reject directory
        shutil.move(input_filename, reject_filename)


def main(
    input_d,
    output_d,
    reject_d,
    fheight=400,
    fwidth=400,
    facePercent=80,
    padUp=False,
    padDown=False,
    padLeft=False,
    padRight=False,
):
    """Crops folder of images to the desired height and width if a face is found

    If input_d == output_d or output_d is None, overwrites all files
    where the biggest face was found.

    Args:
        input_d (str): Directory to crop images from.
        output_d (str): Directory where cropped images are placed.
        reject_d (str): Directory where images that cannot be cropped are
                        placed.
        fheight (int): Height (px) to which to crop the image.
                       Default: 500px
        fwidth (int): Width (px) to which to crop the image.
                       Default: 500px
        facePercent (int) : Percentage of face from height,
                       Default: 50
    Side Effects:
        Creates image files in output directory.

    str, str, (int), (int) -> None
    """
    reject_count = 0
    output_count = 0
    input_files = [
        os.path.join(input_d, f)
        for f in os.listdir(input_d)
        if any(f.endswith(t) for t in INPUT_FILETYPES)
    ]
    if output_d is None:
        output_d = input_d
    if reject_d is None and output_d is None:
        reject_d = input_d
    if reject_d is None:
        reject_d = output_d

    # Guard against calling the function directly
    input_count = len(input_files)
    assert input_count > 0

    # Main loop
    i=True
    k=0
    for input_filename in input_files:
        basename = os.path.basename(input_filename)
        output_filename = os.path.join(output_d, basename)
        reject_filename = os.path.join(reject_d, basename)

        input_img = open_file(input_filename)
        image = None

        # Attempt the crop
        if(k>0):
            i = False
            
        try:
            image = crop(
                input_img,
                i,
                fheight,
                fwidth,
                facePercent,
                padUp,
                padDown,
                padLeft,
                padRight,
            )
            k += 1
        except ImageReadError:
            print("Read error:       {}".format(input_filename))
            continue

        # Did the crop produce an invalid image?
        if isinstance(image, type(None)):
            reject(input_filename, reject_filename)
            print("No face detected: {}".format(reject_filename))
            reject_count += 1
        else:
            output(input_filename, output_filename, image)
            # print("Face detected:    {}".format(output_filename))
            output_count += 1

    # Stop and print status
    print(
        "{} input files, {} faces cropped, {} rejected".format(
            input_count, output_count, reject_count
        )
    )


def autocrop_faces(images_dir):
    print("autocrop")
    faces_dir = os.path.join(images_dir, "faces")
    os.makedirs(faces_dir)
    print("Begin processing images in folder:", faces_dir)
    garbage = os.path.join(images_dir, "garbage")
    os.makedirs(garbage)
    main(
        images_dir,
        faces_dir,
        #r"D:\Deepfake\Training data\FaceForensics++ datasets\manipulated_sequences\Face2Face\c0\garbage",
        garbage,
        facePercent=95
        )
    print("End processing images in folder:", faces_dir)
    return images_dir
