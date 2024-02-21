import scipy.signal as signal
import numpy as np
import pandas as pd

"""
Finds the troughs in the spectrogram for emission identification

Jack Kelley
"""

doublets = data = pd.read_csv('line_list.txt', header=None, delim_whitespace=True)
doublets = doublets.drop(2, axis=1)
doublets = doublets.drop(4, axis=1)
doublets.columns = ('Doublet', 'Blue', 'Red')
doublets = doublets.sort_values(by='Blue', ascending=False)

def identify_troughs(data, prominence, redshift):
    """
    Flips the data and identifies the troughs. Because the dataframe no longer starts at 0, add the first index
    to shift the indexes of the returned values by the same amount.

    TODO: set the min distance to allow a next peak to be greater than the maximum distance between the farthest
    doublet, then do a loop on each peak to find if there is another one with the distance equal to a doublet distance.
    """

    flux_data = data['Flux'].loc[data['Rest Wavelength'] > 1216]
    index_shift = flux_data.index[0]
    flux_data = -flux_data
    min_indexes = signal.find_peaks(flux_data, prominence=prominence)
    min_indexes = min_indexes[0] + index_shift

    return min_indexes, index_shift

def determine_possible_redshifts(data, troughs):
    """
    Pass magnesium troughs to start with
    :param: data the spectrum dataset
    :param: troughs the troughs that will be used to determine redshifts
    :return:
    """
    rest_troughs = data['Rest Wavelength'].iloc(troughs)


