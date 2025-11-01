from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import normalize
from glob import glob
from stardist import fill_label_holes, calculate_extents
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from PIL import Image
from tqdm import tqdm
import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# X_filenames represents the raw images
X_filenames = sorted(glob("train/images/*.*"))

# function to parse arguments given in command line
def parse_args():
    """
    Parses command-line arguments for configuring dataset parameters, model training, and evaluation settings.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Help section for optional commands.")

    parser.add_argument(
        "--total_data", 
        type=int, 
        default=len(X_filenames), 
        help="Total amount of data. Default: total number of images in the images folder."
    )
    parser.add_argument(
        "--dataset_size", 
        type=int, 
        default=(len(X_filenames) - 1), 
        help="Size of the dataset to be used. Must be less than total_data to ensure testing data is available. Default: one less than total_data."
    )
    parser.add_argument(
        "--rays", 
        type=int, 
        default=32, 
        help="Number of rays to be used. Default: 32."
    )
    parser.add_argument(
        "--train_split", 
        type=float, 
        default=0.80, 
        help="Percentage split for training/validation data. Default: 0.80 (80%)."
    )
    parser.add_argument(
        "--testing_size", 
        type=int, 
        default=1, 
        help="Number of testing images. Must be at least 1 to ensure the program runs correctly. Default: 1."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        nargs='+', 
        default=[10], 
        help="Number of training epochs. Accepts a single value or a list (e.g., --epochs 10 50 100 300). Default: 10."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="customModel", 
        help="Name of the model. Default: 'customModel'."
    )

    return parser.parse_args()

# function to read images using Pillow
def read_image(filename, grayscale=False):
    with Image.open(filename) as img:
        if grayscale:
            img = img.convert('L')
        return np.array(img)

# main function
def main(args):
    """
    Main function to process image data, set up training configurations, and train a segmentation model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments specifying dataset size, model parameters, 
                                training configuration, and other options.

    Steps:
        1. Loads image and mask data from disk.
        2. Normalizes and preprocesses images (e.g., normalization, hole filling).
        3. Splits data into training, validation, and testing sets.
        4. Configures the segmentation model with user-defined parameters.
        5. Defines augmentation techniques for training.
        6. Trains the model for specified epochs with data augmentation.
        7. Evaluates model performance on validation and test sets.
        8. Saves model evaluation metrics and visualization plots.

    Outputs:
        - Training, validation, and testing images saved for verification.
        - Model training progress and evaluation metrics.
        - CSV reports with model performance statistics.
        - Trained models saved in the specified directory.
    """

    # Y represents the masks
    Y_filenames = sorted(glob("train/masks/*.*"))

    # Read image files and store them in a list named X
    X = [read_image(x) for x in tqdm(X_filenames, desc="Loading images")]

    # Read mask files and store them in a list named Y
    Y = [read_image(y) for y in tqdm(Y_filenames, desc="Loading masks")]

    # ensures that any random operations are reproducible across all runs
    np.random.seed(42)

    # print statements
    print(f"Total amount of data: {args.total_data}")
    print(f"Dataset_size: {args.dataset_size}")
    print(f"Testing size: {args.testing_size}")
    print(f"Rays: {args.rays}")
    print(f"Training/validation split: {args.train_split}")
    print(f"Epochs: {args.epochs}")
    print("Total number of images:", len(X))
    print("Total number of masks:", len(Y))

    # n_channels are set to 3 indicating that the input images are RGB
    n_channel = 3

    # normalization of images
    axis_norm = (0, 1)
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X, desc="Normalizing images")]

    # Convert masks to integer labels
    Y = [y.astype(np.int32) for y in tqdm(Y, desc="Converting masks to int32")]

    # Fill label holes in masks
    Y = [fill_label_holes(y) for y in tqdm(Y, desc="Filling label holes in masks")]


    ######################################################################################################
    # Uncomment this section if your images are less than the minimum requirement of 256x256 pixels
    # function to pad images the minimum
    # def pad_image(img, target_shape=(256, 256)):
    #     pads = [(0, max(0, target_shape[i]-img.shape[i])) if i < 2 else (0, 0) for i in range(img.ndim)]
    #     return np.pad(img, pads, mode='constant')

    # padding images
    # X = [pad_image(x) for x in X]
    # Y = [pad_image(y) for y in Y]
    ######################################################################################################


    # number of rays to use for the non-maximum suppression in the StarDist training
    # 32 is default
    n_rays = args.rays

    # parameter in StarDist that specifies the step size for the patches extracted from
    # the training images.
    grid = (4, 4)

    # configuration object with model parameters to be used in initilization/training
    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        n_channel_in=n_channel,
    )

    # function for random flips/rotations of the data
    def random_fliprot(img, mask):
        assert img.ndim >= mask.ndim
        axes = tuple(range(mask.ndim))
        perm = tuple(np.random.permutation(axes))
        img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
        mask = mask.transpose(perm)
        for ax in axes:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask

    # function to give data random intensity
    def random_intensity_change(img):
        img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        return img

    # this is the augmenter that will go into the training of the model
    def augmenter(x, y):
        x, y = random_fliprot(x, y)
        x = random_intensity_change(x)
        sig = 0.02 * np.random.uniform(0, 1)
        x = x + sig * np.random.normal(0, 1, x.shape)
        return x, y

    # should be set to the same number as the seed at the beginning of the script
    rng = np.random.default_rng(42)

    # size of dataset to train on
    dataset_size = args.dataset_size

    # amount of total data: change this to suit your needs
    total_data = args.total_data

    # selects random unique images for testing from the total images
    test_indices = rng.choice(total_data, size=args.testing_size, replace=False)

    # get your test datasets
    X_test, Y_test = [X[i] for i in test_indices], [Y[i] for i in test_indices]

    # remaining images are the ones that aren't in the test set
    all_remaining_indices = list(set(range(total_data)) - set(test_indices))

    # select subset from the remaining based on dataset_size
    selected_indices = rng.choice(all_remaining_indices, size=dataset_size, replace=False)

    # determine sizes for training and validation
    # % of the dataset_size
    n_train_split = args.train_split
    n_train = int(n_train_split * dataset_size)

    # split the selected data into training and validation
    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:]

    # get your training and validation datasets
    X_train, Y_train = [X[i] for i in train_indices], [Y[i] for i in train_indices]
    X_val, Y_val = [X[i] for i in val_indices], [Y[i] for i in val_indices]

    # prints amount of images in training, validation, and testing to verify
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # where the model will be saved
    base_dir = "models"

    # make a new directory for the dataset size
    dataset_dir = os.path.join(base_dir, f'datasize_{dataset_size}')
    os.makedirs(dataset_dir, exist_ok=True)

    # function to save training, validation, and testing data in a grid to a PNG file
    def save_images_to_file(images, filename, title):
        # specify the dimensions of the subplot grid
        n = len(images)
        cols = int(math.sqrt(n))  # assuming you want a square grid, change this as per your requirements
        rows = int(math.ceil(n / cols))

        # create a new figure with specified size
        fig = plt.figure(figsize=(20, 20))  # adjust as needed

        # set title
        plt.title(title, fontsize=40)  # adjust font size as needed

        # iterate over each image and add it to the subplot
        for i in range(n):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i])  # using gray colormap as these are grayscale images
            ax.axis('off')  # to remove axis

        # adjust layout and save the figure
        fig.tight_layout()  # adjust layout so labels do not overlap
        fig.savefig(filename, dpi=600)

    # this section saves the training, validation, and testing data in separate images to
    # see the data the program selected using the function from above
    # saving the training images
    training_filename = os.path.join(dataset_dir, 'training_images.png')  # define the path and name for your image
    save_images_to_file(X_train, training_filename, "Training Images")

    # saving the validation images
    validation_filename = os.path.join(dataset_dir, 'validation_images.png')  # define the path and name for your image
    save_images_to_file(X_val, validation_filename, "Validation Images")

    # saving the testing images
    testing_filename = os.path.join(dataset_dir, 'testing_images.png')  # define the path and name for your image
    save_images_to_file(X_test, testing_filename, "Testing Images")

    # function to evaluate and save csv files with stats
    def evaluate_and_save(model, X_data, Y_data, data_type='validation'):

        # prediction
        Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(X_data)]

        # evaluation
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset(Y_data, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

        # plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        metrics = ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality')
        counts = ('fp', 'tp', 'fn')

        for m in metrics:
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in counts:
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()

        # save figure
        figure_filename = os.path.join(model.basedir, model.name, f"{data_type}_plots.png")
        fig.savefig(figure_filename, dpi=300)

        # save CSV
        filename = os.path.join(model.basedir, model.name, f'{data_type}_stats.csv')
        fieldnames = list(stats[0]._asdict().keys())

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in stats:
                writer.writerow(entry._asdict())

        return stats

    # number of epochs
    epochs = args.epochs
    
    # main training loop
    for i in epochs:

        # naming the model
        model_name = args.model_name + "_" + str(args.dataset_size) + '_epochs_' + str(i)

        # instantiate the model with custom parameters
        model = StarDist2D(conf, name=model_name, basedir=dataset_dir)

        # calculates the average size of labeled objects in mask images
        median_size = calculate_extents(list(Y_train), np.median)

        # refers to how much the network can "see" the image in a single pass
        fov = np.array(model._axes_tile_overlap('YX'))
        print(f"Median object size:      {median_size}")
        print(f"Network field of view :  {fov}")
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")
            print("Adjust the variable \"grid\" to be higher than (2,2).")
        # IMPORTANT: MAKE SURE THE NETWORK FOV IS BIGGER THAN OBJECT SIZE OTHERWISE IT
        # CAN CAUSE THE NETWORK TO STRUGGLE TO DETECT THE OBJECTS PROPERLY
        # THIS CAN LEAD TO PARTIAL SEGMENTATIONS OR MISSED DETECTIONS

        # epochs based on where i is in the list of epochs
        epochs = i

        # code to train the model based on the data given
        model.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, augmenter=augmenter)

        # optimizing thresholds for validation data
        model.optimize_thresholds(X_val, Y_val)

        # evaluation of validation data
        stats_val = evaluate_and_save(model, X_val, Y_val, 'validation')

        # evaluation of testing data
        stats_test = evaluate_and_save(model, X_test, Y_test, 'test')

    print("Training is complete.")

if __name__ == "__main__":
    # loading arguments
    args = parse_args()
    main(args)