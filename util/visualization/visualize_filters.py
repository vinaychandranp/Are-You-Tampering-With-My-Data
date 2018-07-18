"""
This script generates visualizations of the weights of intermediate layers of CNNs.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


def normalize_array(arr):
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    arr = arr * 255
    return arr


def resize_as_img(arr):
    img = Image.fromarray(arr.astype('uint8'))
    img = img.resize(size=(2000, 2000), resample=Image.BICUBIC)
    return img


def save_images_to_folder(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, image in enumerate(images):
        img = Image.fromarray(image.astype('uint8'))
        img.save(os.path.join(folder, '{}.png'.format(i)))
    return


def main(args):
    """
    Main routine of script to generate weight heatmaps.
    Parameters
    ----------
    args : argparse.Namespace
        contains all arguments parsed from input

    Returns
    -------
    None

    """
    state_dict = torch.load(args.checkpoint)['state_dict']
    if args.layer is None:
        print(state_dict.keys())
        print('Enter the name of the layer you would like to visualize')
        layer_name = input()
    else:
        layer_name = args.layer

    data = state_dict[layer_name].cpu()
    data_inv = state_dict[layer_name].cpu() * -1

    img = make_grid(data, nrow=4, scale_each=True, normalize=True).numpy().transpose(1, 2, 0) * 255
    img_inv = make_grid(data_inv, nrow=4, scale_each=True, normalize=True).numpy().transpose(1, 2, 0) * 255

    data = data.numpy()
    for i in range(len(data)):
        data[i] = normalize_array(data[i])

    data_inv = data_inv.numpy()
    for i in range(len(data_inv)):
        data_inv[i] = normalize_array(data_inv[i])

    data = np.transpose(data, (0, 2, 3, 1))
    data_inv = np.transpose(data_inv, (0, 2, 3, 1))

    save_images_to_folder(data, os.path.join(args.output, 'output'))
    save_images_to_folder(data_inv, os.path.join(args.output, 'output_inv'))
    resize_as_img(img).save(os.path.join(args.output, 'output', 'grid.png'))
    resize_as_img(img_inv).save(os.path.join(args.output, 'output_inv', 'grid.png'))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='path to latest checkpoint')
    parser.add_argument('--layer',
                        type=str,
                        default=None,
                        help='layer to visualize the activations from')
    parser.add_argument('--output',
                        type=str,
                        default='./',
                        help='path to folder where the output image should be saved')

    args = parser.parse_args()

    main(args)
