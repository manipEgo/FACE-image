import os
import json
import numpy as np
from typing import List

def load_folder_information(path: str) -> dict:
    '''
    Read folder information from json file in the folder

    Parameters:
    -----------
        path: `str`, the path to the folder

    Returns:
    --------
        `dict`, the folder information
    '''
    file_name = "info.json"
    required_items = ["origin", "mask_prefix"]
    # check if the path exists & is a directory
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory")
    # check if the folder information file exists
    file_path = os.path.join(path, file_name)
    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path} is not a valid file")
    # read the folder information
    info = {}
    with open(file_path, "r") as file:
        info = json.load(file)
    # check if the required items are present
    for item in required_items:
        if item not in info:
            raise ValueError(f"{item} is not present in {file_path}")

def load_image(path: str) -> np.array:
    '''
    Load an image from a given path

    Parameters:
    -----------
        path: `str`, the path to the image

    Returns:
    --------
        `np.array`, the image
    '''
    return np.load(path)

def image_MSE(origin: np.array, generated: np.array, mask: np.array) -> float:
    '''
    Calculate the mean squared error between the origin and generated images, with the mask applied

    Parameters:
    -----------
        origin: `np.array`, the original image
        generated: `np.array`, the generated image
        mask: `np.array`, the mask to be applied

    Returns:
    --------
        `float`, the mean squared error
    '''
    diff = origin - generated
    diff = np.multiply(diff, mask)
    return np.mean(np.square(diff))

def sequence_spectrum(sequence: np.array) -> np.array:
    '''
    Calculate the spectrum of a sequence

    Parameters:
    -----------
        sequence: `np.array`, the sequence to be analyzed

    Returns:
    --------
        `np.array`, the spectrum of the sequence, with the real part only
    '''
    return np.fft.fft(sequence).real

def spectrum_scores(spectrum: np.array):
    pass