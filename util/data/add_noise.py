import argparse
import os
import cv2
import numpy as np

from util.misc import get_all_files_in_folders_and_subfolders

def change_ext(path):
    path = path.split('.')[0] + '.png'
    return path


def apply_mask_to_image(mask, image_path):
    img = cv2.imread(image_path)
    img[mask[:, 0], mask[:, 1], 0] = 0
    os.remove(image_path)
    image_path = change_ext(image_path)
    cv2.imwrite(image_path, img)
    return None


def main(args):
    files = get_all_files_in_folders_and_subfolders(args.folder)

    img_shape = cv2.imread(files[0]).shape

    # Seed the RNG generator
    np.random.seed(seed=42)

    # Make a random noise mask
    if args.num_pixels is None:
        num_points_to_scramble = int(0.005 * img_shape[0] * img_shape[1])
    else:
        num_points_to_scramble = args.num_pixels
    print('Number of scrambled points: {}'.format(num_points_to_scramble))
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
    parser.add_argument('--num-pixels',
                        default=None,
                        help='number of pixels to corrupt. IF not specified, 0.5% pixels of the image are corrupted',
                        type=int)
    args = parser.parse_args()

    main(args)
