import numpy as np

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