"""
This script allows for creation of a point cloud (2D) dataset.

Structure of the dataset:

Split folders
-------------
'args.dataset-folder' has to point to where the dataset will be saved.

Example:

        ~/../../data

Classes folders
---------------
The splits will have different classes in a separate folder with the class
name. The file name can be arbitrary (e.g does not have to be 0-* for classes 0 of MNIST).
Example:

    train/dog/whatever.png
    train/dog/you.png
    train/dog/like.png

    train/cat/123.png
    train/cat/nsdf3.png
    train/cat/asd932_.png

@author: Michele Alberti
"""

# Utils
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Distribution options:
distribution_options = ['diagonal', 'circle']


def diagonal(size):
    """
    Samples are generated in a grid fashion (np.linspace) and then draw a diagonal line on x=y
    2 classes.

    Parameters
    ----------
    :param size: int
        The total number of points in the dataset.
    :return:
        train, val, test where each of them has the shape [n,3]. Each row is (x,y,label)
    """

    # Generate data
    samples = np.array([(x, y, 0 if x > y else 1)
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return split_data(samples)


def circle(size):
    """
    Samples are generated in a grid fashion (np.linspace) and then draw a circle on x*x + y*y > 0.5
    2 classes.

    Parameters
    ----------
    :param size: int
        The total number of points in the dataset.
    :return:
        train, val, test where each of them has the shape [n,3]. Each row is (x,y,label)
    """

    # Generate data
    samples = np.array([(x, y, 0 if x * x + y * y > 0.5 else 1)
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return split_data(samples)


def split_data(samples):
    """
    Split the given samples array into train validation and test sets with ratio 6, 2, 2

    Parameters
    ----------
    :param samples: np.array(n,m+1)
        The samples to be split: n is the number of samples, m is the number of dimensions and the +1 is the label

    :return:
        train, val, test where each of them has the shape [n,3]. Each row is (x,y,label)
    """
    # Split it
    train, tmp, label_train, label_tmp = train_test_split(samples[:, 0:2], samples[:, 2], test_size=0.4,
                                                          random_state=42)
    val, test, label_val, label_test = train_test_split(tmp, label_tmp, test_size=0.5, random_state=42)

    # Return the different splits by selecting x,y from the data and the relative label
    return np.array([[a[0], a[1], b] for a, b in zip(train, label_train)]), \
           np.array([[a[0], a[1], b] for a, b in zip(val, label_val)]), \
           np.array([[a[0], a[1], b] for a, b in zip(test, label_test)])


def get_data(distribution, size):
    """
    Return the train, val and test splits according to the distribution chosen.

    Parameters
    ----------
    :param distribution: enum \in distribution_options
        The chosen distribution
    :param size:
        The total number of samples in the dataset. It is advised to use a squared number if a grid-fashion distribution
        is chosen (like 16, 25, 100, ... )
    :return:
        train, val and test splits
    """
    return {
        'diagonal': diagonal,
        'circle': circle,
    }[distribution](size)


if __name__ == "__main__":
    ###############################################################################
    # Argument Parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script allows for creation of a validation set from the training set')

    parser.add_argument('--dataset-folder',
                        help='location of the dataset on the machine e.g root/data',
                        required=True,
                        type=str)

    parser.add_argument('--distribution',
                        help='Kind of distribution of the points',
                        choices=distribution_options,
                        required=True,
                        type=str)

    parser.add_argument('--size',
                        help='Total amount of samples.',
                        type=int,
                        default=100)

    args = parser.parse_args()

    ###############################################################################
    # Getting the data
    train, val, test = get_data(args.distribution, args.size)

    ###############################################################################
    # Preparing the folders structure

    # Sanity check on the dataset folder
    if not os.path.isdir(args.dataset_folder):
        print("Dataset folder not found in the args.dataset_folder={}".format(args.dataset_folder))
        sys.exit(-1)

    # Creating the folder for the dataset
    dataset_dir = os.path.join(args.dataset_folder, 'pc_' + args.distribution)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    # Creating the folders for the splits
    train_dir = os.path.join(dataset_dir, 'train')
    os.makedirs(train_dir)

    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(val_dir)

    test_dir = os.path.join(dataset_dir, 'test')
    os.makedirs(test_dir)

    ###############################################################################
    # Save splits on csv format with n rows where each row is (x,y,label)
    pd.DataFrame(train).to_csv(os.path.join(train_dir, 'data.csv'), index=False, header=False)
    pd.DataFrame(val).to_csv(os.path.join(val_dir, 'data.csv'), index=False, header=False)
    pd.DataFrame(test).to_csv(os.path.join(test_dir, 'data.csv'), index=False, header=False)

    ###############################################################################
    # Run the analytics
    mean = np.mean(train[:, 0:-1], 0)
    std = np.std(train[:, 0:-1], 0)

    # Save results as CSV file in the dataset folder
    df = pd.DataFrame([mean, std])
    df.index = ['mean[RGB]', 'std[RGB]']
    df.to_csv(os.path.join(dataset_dir, 'analytics.csv'), header=False)

    print('done')
