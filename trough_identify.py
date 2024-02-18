from scipy.signal import find_peaks
import numpy as np

"""
Finds the troughs in the spectrogram for emission identification

Jack Kelley
"""


def denoise(data, window_size):
    """
    Uses a discrete convolution to smooth the data
    :param data: data to be smoothed
    :param window_size: the size of the rolling window
    :return: the smoothed data
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def identify_absorption(data, prominence):
    """
    Flips the data and identifies the troughs
    """
    data = -data
    return find_peaks(data, prominence=prominence)
