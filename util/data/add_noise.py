import argparse

import cv2
import numpy as np

from util.misc import get_all_files_in_folders_and_subfolders


def apply_mask_to_image(mask, image):
    img = cv2.imread(image)

    img[mask[:, 0], mask[:, 1], 0] = 0

    cv2.imwrite(image, img)

    return None


def main(args):
    files = get_all_files_in_folders_and_subfolders(args.folder)

    img_shape = cv2.imread(files[0]).shape

    # Seed the RNG generator
    np.random.seed(seed=42)

    # Make a random noise mask
    num_points_to_scramble = int(0.005 * img_shape[0] * img_shape[1])
    mask = np.random.randint(low=0, high=img_shape[0], size=(num_points_to_scramble, 2))



    for file in files:
        apply_mask_to_image(mask, file)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to add noise to images')

    parser.add_argument('folder',
                        help='path to a folder of images to add noise to',
                        type=str)
    args = parser.parse_args()

    main(args)
