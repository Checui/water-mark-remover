import numpy as np


def construct_dataset(wm_images: np.ndarray, nowm_images: np.ndarray):
    """Returns X, y (0=watermarked, 1=non-watermarked)"""
    x = np.concatenate((wm_images, nowm_images))
    y = np.concatenate((np.zeros(wm_images.shape[0]), np.ones(nowm_images.shape[0])))
    return x, y
