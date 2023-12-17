"""Funciones extraídas del notebook de Kaggle (TODO: poner link)"""
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def take_file_name(filedir):  # remove just file name from directory and return
    # filename = np.array(filedir.split('/'))[-1].split('.')[0] # take out the name, isolate the jpeg, then return the name
    filename = np.array(filedir.split("/"))[
        -1
    ]  # take out the name, then return the name
    # print(filename)
    return filename


def match_file_names(watermarkedarr, nonwatermarkedarr, dname_wm, dname_nwm):
    sortedwmarr = np.array([])
    sortednwmarr = np.array([])

    wmarr = list(watermarkedarr)
    nwmarr = list(nonwatermarkedarr)

    length = (
        len(watermarkedarr)
        if len(watermarkedarr) >= len(nonwatermarkedarr)
        else len(nonwatermarkedarr)
    )

    for pos in range(length):
        try:
            if length == len(watermarkedarr):  # more images in watermarked array
                exist_nwm = nwmarr.index(wmarr[pos])
                sortedwmarr = np.append(
                    sortedwmarr, dname_wm + watermarkedarr[pos]
                )  # this is the iterable
                sortednwmarr = np.append(
                    sortednwmarr, dname_nwm + nonwatermarkedarr[exist_nwm]
                )  # this is the match
            elif length == len(
                nonwatermarkedarr
            ):  # more images in nonwatermarked array
                exist_wm = wmarr.index(nwmarr[pos])
                sortedwmarr = np.append(
                    sortedwmarr, dname_wm + watermarkedarr[exist_wm]
                )  # this is the match
                sortednwmarr = np.append(
                    sortednwmarr, dname_nwm + nonwatermarkedarr[pos]
                )  # this is the iterable
        except ValueError:
            continue
    return sortedwmarr, sortednwmarr


def take_file_name(file_path):
    # Assuming this function extracts just the file name from a file path
    return os.path.basename(file_path)


def read_image_names(path):
    image_names = np.array([])
    for _, _, files in os.walk(path, topdown=True):
        for file in files:
            image_names = np.append(image_names, take_file_name(file))
    return image_names


def prepare_image_datasets(
    train_wm_path, train_nwm_path, valid_wm_path, valid_nwm_path
):
    # Read and sort training dataset filenames
    train_watermarked = read_image_names(train_wm_path)
    train_nonwatermarked = read_image_names(train_nwm_path)
    train_watermarked_sorted, train_nonwatermarked_sorted = match_file_names(
        train_watermarked, train_nonwatermarked, train_wm_path, train_nwm_path
    )

    # Read and sort validation dataset filenames
    valid_watermarked = read_image_names(valid_wm_path)
    valid_nonwatermarked = read_image_names(valid_nwm_path)
    valid_watermarked_sorted, valid_nonwatermarked_sorted = match_file_names(
        valid_watermarked, valid_nonwatermarked, valid_wm_path, valid_nwm_path
    )

    return {
        "train": {
            "watermarked": train_watermarked_sorted,
            "nonwatermarked": train_nonwatermarked_sorted,
        },
        "valid": {
            "watermarked": valid_watermarked_sorted,
            "nonwatermarked": valid_nonwatermarked_sorted,
        },
    }


def load_and_resize_images(file_paths, width, height):
    """Loads and resize images to the specified width and height.

    Parameters:
        file_paths (list of str): List of file paths to the images.
        width (int): Width to resize the images to.
        height (int): Height to resize the images to.

    Returns:
        np.array: Array of resized images.
    """
    dim = (width, height)
    data = []
    for image_path in file_paths:
        try:
            img_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            resized_arr = cv2.resize(img_arr, dim)
            data.append(resized_arr)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return np.array(data)


def data_augmentation(input_image):
    return random_contrast(random_brightness(input_image)).numpy()


def random_flip(pic):
    return tf.image.random_flip_up_down(tf.image.random_flip_left_right(pic, 1), 1)


def random_brightness(pic):
    return tf.image.random_brightness(pic, random.uniform(0.01, 0.2), 1)


def random_contrast(pic):
    return tf.image.random_contrast(pic, 0.2, 0.7, 1)


def create_pixel_arr(files, width, height):
    data = []
    for image in files:
        try:  # take each image and use imread to get the pixel values in a matrix
            img_arr = cv2.imread(image, cv2.IMREAD_COLOR)
            resized_arr = cv2.resize(
                img_arr, (width, height)
            )  # rescale the image so every image is of the same dimension
            data.append(resized_arr)  # add the matrix of pixel values
        except Exception as e:
            print(e)  # some error thrown in imread or resize
    return np.array(data)


def load_and_prepare_data(
    train_wm_path,
    train_nwm_path,
    valid_wm_path,
    valid_nwm_path,
    train_size=0.8,
    width=128,
    height=128,
    use_data_augmentation=True,
):
    """Returns X_train, X_test, y_train, y_test, X_val, y_val"""
    # Read and match training data filenames
    tp_watermarked = read_image_names(train_wm_path)
    tp_nonwatermarked = read_image_names(train_nwm_path)
    tp_watermarked_sorted, tp_nonwatermarked_sorted = match_file_names(
        tp_watermarked, tp_nonwatermarked, train_wm_path, train_nwm_path
    )

    # Read and match validation data filenames
    vp_watermarked = read_image_names(valid_wm_path)
    vp_nonwatermarked = read_image_names(valid_nwm_path)
    vp_watermarked_sorted, vp_nonwatermarked_sorted = match_file_names(
        vp_watermarked, vp_nonwatermarked, valid_wm_path, valid_nwm_path
    )

    # Load and resize images
    train_wms_pixVals = create_pixel_arr(tp_watermarked_sorted, width, height)
    train_nwms_pixVals = create_pixel_arr(tp_nonwatermarked_sorted, width, height)
    X_val = create_pixel_arr(vp_watermarked_sorted, width, height)
    y_val = create_pixel_arr(vp_nonwatermarked_sorted, width, height)

    # Split training data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        train_wms_pixVals, train_nwms_pixVals, train_size=train_size, random_state=1
    )

    # Data Augmentation
    if use_data_augmentation:
        data_augmented_X = [data_augmentation(img) for img in X_train]
        data_augmented_y = [data_augmentation(img) for img in y_train]

        # Append augmented data to training data
        X_train = np.append(X_train, data_augmented_X, axis=0)
        y_train = np.append(y_train, data_augmented_y, axis=0)

    # Normalize the data
    X_train = X_train / 255
    y_train = y_train / 255
    X_test = X_test / 255
    y_test = y_test / 255

    # Return prepared data
    return X_train, X_test, y_train, y_test, X_val, y_val
