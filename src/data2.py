from __future__ import print_function

from pathlib import Path

import cv2
import numpy as np

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
        # processed = cv2.resize(reshaped[0], (ds_img_cols, ds_img_rows), interpolation=cv2.INTER_CUBIC)
        # output_img[i] = np.array([processed]).reshape(ds_img_rows, ds_img_cols, input_img.shape[-1])
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return output_img


def create_train_data(split='train'):
    """
    Generate training data numpy arrays and save them into the project path
    """

    ds_img_rows = 246
    ds_img_cols = 288

    input_dir_1 = 'input1'
    input_dir_2 = 'input2'
    label_dir = 'label'

    images = []
    masks = []

    scale_ratio_r = 1
    scale_ratio_c = 1
    if ds_img_rows < 256:
        scale_ratio_r = 256 / ds_img_rows
    if ds_img_cols < 256:
        scale_ratio_c = 256 / ds_img_cols
    scale_ratio = max(scale_ratio_r, scale_ratio_c)
    if scale_ratio > 1:
        new_width = int(ds_img_cols * scale_ratio)
        new_height = int(ds_img_rows * scale_ratio)

    def open_patch(root, patch, path):
        img1 = cv2.imread(str(patch))
        img2 = cv2.imread(str(Path(root, input_dir_2, path)))

        if scale_ratio > 1:
            img1 = cv2.resize(img1, (new_width, new_height))
            img2 = cv2.resize(img2, (new_width, new_height))

        img = np.concatenate((img1, img2), axis=-1)
        img = img.reshape((n_channel, new_height, new_width))
        images.append(img)

        mask = cv2.imread(str(Path(root, label_dir, path)), cv2.IMREAD_GRAYSCALE)
        if scale_ratio > 1:
            mask = cv2.resize(mask, (new_width, new_height))
        mask = 255 - mask.reshape(1, new_height, new_width)
        masks.append(mask)

    def open_set(set_name):
        root = DATA_PATH + set_name
        for regionOrPatch in Path(root, input_dir_1).iterdir():
            if regionOrPatch.is_dir():
                for patch in regionOrPatch.iterdir():
                    open_patch(root, patch, Path(regionOrPatch.stem, patch.name))
            else:
                open_patch(root, regionOrPatch, regionOrPatch.name)

    open_set(split)

    images = np.asarray(images)
    masks = np.asarray(masks)

    print(images.shape)
    print(masks.shape)

    # reduce num of samples for debugging
    # images = images[:30]
    # masks = masks[:30]

    np.save(f'../../data/imgs_{split}.npy', images)
    np.save(f'../../data/imgs_mask_{split}.npy', masks)


def load_data(split='train'):
    """
    Load training data from project path
    :return: [X_train, y_train] numpy arrays containing the training data and their respective masks.
    """
    print(f"\nLoading {split} data...\n")
    X_train = np.load(f'../../data/imgs_{split}.npy')
    y_train = np.load(f'../../data/imgs_mask_{split}.npy')

    X_train = preprocessor(X_train)
    y_train = preprocessor(y_train)

    X_train = X_train.astype('float32')

    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)  # std for data normalization

    X_train -= mean
    X_train /= std

    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]
    y_train[y_train < 0.5] = 0.
    y_train[y_train >= 0.5] = 1.
    return X_train, y_train


if __name__ == '__main__':
    create_train_data('train')
    create_train_data('val')
    create_train_data('test')
