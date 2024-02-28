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
doublets = doublets.reset_index(drop=True)


def identify_troughs(passed_data, prominence, redshift):
    """
    Flips the data and identifies the troughs. Because the dataframe no longer starts at 0, add the first index
    to shift the indexes of the returned values by the same amount.

    TODO: set the min distance to allow a next peak to be greater than the maximum distance between the farthest
    doublet, then do a loop on each peak to find if there is another one with the distance equal to a doublet distance.
    """

    flux_data = passed_data['Flux'].loc[passed_data['Rest Wavelength'] > 1216]
    index_shift = flux_data.index[0]  # starting index is the point where rest wavelength is 1216
    flux_data = -flux_data
    min_indexes = signal.find_peaks(flux_data, prominence=prominence)
    min_indexes = min_indexes[0] + index_shift

    return min_indexes, index_shift


def determine_possible_redshifts(passed_data, troughs):
    """
    Pass magnesium troughs to start with
    :param: passed_data the spectrum dataset
    :param: troughs the troughs that will be used to determine redshifts
    :return:
    """

    ## get the troughs that are MGII candidates
    # data with wavelengths near MgII emission
    possible_mg2 = passed_data['Observed Wavelength'][passed_data['Rest Wavelength'] > 2600]
    # cutoff trough indexes to minimum possible_mg2
    troughs = troughs[troughs >= possible_mg2.index[0]]
    possible_mg2 = possible_mg2.loc[troughs]

    # make list of possible redshifts
    intervening_z = []
    for trough in possible_mg2:
        """
        Right now I am only checking the MgII to generate possible redshifts, so
        I use iloc[0] to pull the value for MgII as it is the first entry in the doublets dataframe.
        """
        temp_z_blue = (trough / float(doublets['Blue'].iloc[0])) - 1
        temp_z_red = (trough / float(doublets['Red'].iloc[0])) - 1
        intervening_z.append(temp_z_blue)
        intervening_z.append(temp_z_red)

    return intervening_z


def match_pairs(passed_data, trough_indexes, redshifts):
    """
    :param: passed_data the spectrum dataset
    :param: troughs
    :param: redshifts the possible redshifts to search
    :return:
    """
    # data with wavelengths near MgII emission
    possible_troughs = passed_data['Observed Wavelength'][passed_data['Rest Wavelength'] > 2600]
    # cutoff trough indexes to minimum possible_mg2
    trough_indexes = trough_indexes[trough_indexes >= possible_troughs.index[0]]
    possible_troughs = possible_troughs.loc[trough_indexes]
    possible_troughs = possible_troughs.reset_index()

    # create dictionary of possible systems
    tagged_doublets = {}
    for z in redshifts:
        tagged_doublets[z] = []

    for i in range(len(possible_troughs)):
        # unpack to get the separate graph index (x-value) and observed wavelength
        index = possible_troughs['index'].iloc[i]
        observed_wavelength = possible_troughs['Observed Wavelength'].iloc[i]
        for z in redshifts:
            """
            Same as in determine_possible_redshifts(), right now we are only checking MgII so its value is 
            left in place and not looped for the time being.
            """
            # get the wavelengths of the theoretical matching troughs
            match_blue = float(doublets['Blue'].iloc[0] * (z + 1))
            match_red = float(doublets['Red'].iloc[0] * (z + 1))

            # save time by only checking troughs that are bluer than the maximum possible matching trough
            checked_troughs = possible_troughs[possible_troughs['Observed Wavelength'] < match_red]

            for j in range(len(checked_troughs)):

                # unpack
                checked_index = checked_troughs['index'].iloc[j]
                checked_observed_wavelength = checked_troughs['Observed Wavelength'][j]

                if ((np.isclose(float(checked_observed_wavelength), match_blue, atol=0.1))
                        or (np.isclose(float(checked_observed_wavelength), match_red, atol=0.1))):

                    # if the trough isn't already in the list, and it isn't matching itself, add it
                    if observed_wavelength != checked_observed_wavelength:
                        tagged_doublets[z].append((index, checked_index))

    # remove redshifts that have no doublets
    for z in list(tagged_doublets):
        # if the tagged doublet is empty, equivalent to if tagged_doublets[z] == []
        if not tagged_doublets[z]:
            del tagged_doublets[z]

    return tagged_doublets
