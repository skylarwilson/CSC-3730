from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import normalize
from glob import glob
from stardist import random_label_cmap
from stardist.models import StarDist2D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from PIL import Image

# displays original pixels without any smoothing or interpolation
matplotlib.rcParams["image.interpolation"] = "none"

# set a random seed for reproducibility
np.random.seed(42)

# generates a random color map for labeled regions
lbl_cmap = random_label_cmap()

# directory containing images for prediction
image_dir = "predictions"

# sorts files by name
file_names = sorted(os.listdir(image_dir))

# selects all .tif images
X = sorted(glob("predictions/*.*"))

# read images using Pillow
X = [np.array(Image.open(f)) for f in X]

# load the specific model you created from the directory of your models
model = StarDist2D(None, name="customModel_6_epochs_20", basedir="models/datasize_6")

# function for prediction
def prediction(model, i, show_dist=True):

    # Normalize per-channel (C not in axis => per-channel percentiles)
    axis_norm = (0, 1)

    # normalize the image for better prediction quality
    img = normalize(X[i], 1, 99.8, axis=axis_norm)

    # uses CPU for prediction - can run into memory problems otherwise
    with tf.device("/cpu:0"):
        labels, details = model.predict_instances(
            img,
            axes="YXC"
            )

    # count the number of unique labels (subtract 1 to exclude the background label 0)
    # num_objects = len(np.unique(labels)) - 1
    num_objects = int(labels.max())
    # get the file name for the current image
    file_name = file_names[i]

    # get the actual count from the file name
    # for this to work, your files need to be labeled in this way:
    # ###count.*
    actual_count = int(file_name.split("count.png")[0])

    # initialize a plot of specified size
    plt.figure(figsize=(13, 10))

    # checks if the image is grayscale or color for visualization
    img_show = img if img.ndim == 2 else img[..., 0]

    # display the segmented labels over the original image
    plt.imshow(img)#, cmap="gray")
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)

    # add the title from the first image
    plt.title(f"Predicted Objects: {num_objects}\nActual Objects: {actual_count}", fontsize=16)

    # hide axes
    plt.axis("off")
    
    # adjust layout
    plt.tight_layout()
    
    # save the figure to your specified directory
    plt.savefig(f"predictions/prediction_test{i}.png", dpi=500)
    plt.close()

# starts the process of predicting
# will predict for each image in X
for i in range(len(X)):
    prediction(model, i)

print("Prediction complete.")