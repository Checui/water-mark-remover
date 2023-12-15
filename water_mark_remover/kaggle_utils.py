"""Funciones extraídas del notebook de Kaggle (TODO: poner link)"""
import os
import random

import cv2
import numpy as np
import tensorflow as tf


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


def data_augmentation(inputImage):
    return randomContrast(randomBrightness(inputImage)).numpy()


def randomFlip(pic):
    return tf.image.random_flip_up_down(tf.image.random_flip_left_right(pic, 1), 1)


def randomBrightness(pic):
    return tf.image.random_brightness(pic, random.uniform(0.01, 0.2), 1)


def randomContrast(pic):
    return tf.image.random_contrast(pic, 0.2, 0.7, 1)
