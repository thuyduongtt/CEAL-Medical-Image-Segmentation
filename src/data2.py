from __future__ import print_function

import os

import cv2
import numpy as np
from pathlib import Path

from constants import *

DATA_PATH = '../../data/QB/'


def preprocessor(input_img):
    """
    Resize input images to constants sizes
    :param input_img: numpy array of images
    :return: numpy array of preprocessed images
    """
    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        # reshaped = input_img[i].reshape(input_img.shape[-1], input_img.shape[1], input_img.shape[2])
        # processed = cv2.resize(reshaped[0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        # output_img[i] = np.array([processed]).reshape(img_rows, img_cols, input_img.shape[-1])
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return output_img


def create_train_data():
    """
    Generate training data numpy arrays and save them into the project path
    """

    image_rows = 256
    image_cols = 256
    n_channel = 6

    input_dir_1 = 'input1'
    input_dir_2 = 'input2'
    label_dir = 'label'

    images = []
    masks = []

    def open_patch(root, patch, path):
        img1 = cv2.imread(str(patch))
        img2 = cv2.imread(str(Path(root, input_dir_2, path)))
        img = np.concatenate((img1, img2), axis=-1)
        img = img.reshape(n_channel, image_rows, image_cols)
        images.append(img)

        mask = cv2.imread(str(Path(root, label_dir, path)), cv2.IMREAD_GRAYSCALE)
        mask = mask.reshape(1, image_rows, image_cols)
        masks.append(mask)

    def open_set(set_name):
        root = DATA_PATH + set_name
        for regionOrPatch in Path(root, input_dir_1).iterdir():
            if regionOrPatch.is_dir():
                for patch in regionOrPatch.iterdir():
                    open_patch(root, patch, Path(regionOrPatch.stem, patch.name))
            else:
                open_patch(root, regionOrPatch, regionOrPatch.name)

    open_set('train')
    open_set('test')
    open_set('val')

    images = np.asarray(images)
    masks = np.asarray(masks)

    print(images.shape)
    print(masks.shape)

    np.save('../../data/imgs_train.npy', images)
    np.save('../../data/imgs_mask_train.npy', masks)


def load_train_data():
    """
    Load training data from project path
    :return: [X_train, y_train] numpy arrays containing the training data and their respective masks.
    """
    print("\nLoading train data...\n")
    X_train = np.load('../../data/imgs_train.npy')
    y_train = np.load('../../data/imgs_mask_train.npy')

    X_train = preprocessor(X_train)
    y_train = preprocessor(y_train)

    X_train = X_train.astype('float32')

    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)  # std for data normalization

    X_train -= mean
    X_train /= std

    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]
    return X_train, y_train


if __name__ == '__main__':
    create_train_data()
