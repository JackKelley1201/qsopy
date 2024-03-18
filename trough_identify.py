import scipy.signal as signal
import numpy as np
import pandas as pd

"""
Finds the troughs in the spectrogram for emission identification

Jack Kelley
"""

doublets = data = pd.read_csv('line_list_short.txt', header=None, delim_whitespace=True)
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

    # get the troughs that are MGII candidates
    # data with wavelengths near MgII emission
    possible_mg2 = passed_data['Observed Wavelength'][passed_data['Rest Wavelength'] > 2000]
    # cutoff trough indexes to minimum possible_mg2
    troughs = troughs[troughs >= possible_mg2.index[0]]
    possible_mg2 = possible_mg2.loc[troughs]

    # make list of possible redshifts
    intervening_z = []
    for trough in possible_mg2:
        """
        Right now I am only checking the MgII to generate possible redshifts, so
        I use iloc[0] to pull the value for MgII as it is the first entry in the doublets dataframe.
        
        This gets the redshifts assuming that each trough is the blue one. There is no need to get the red ones,
        as the red ones would just be duplicates in a different order.
        """
        temp_z = (trough / float(doublets['Blue'].iloc[0])) - 1
        intervening_z.append(temp_z)

    return intervening_z


def match_mgii(passed_data, trough_indexes, redshifts):
    """
    The concept for this function is as follows: take each possible redshift generated by assuming a trough is a blue
    trough in a doublet. Then calculate the theoretical wavelength of the matching red trough and compare for each
    trough and redshift. If a checked trough is the same as the theoretical red trough, put the blue and red trough into
    a tuple and assign it to that redshift in a dictionary.

    :param: passed_data the spectrum dataset
    :param: troughs
    :param: redshifts the possible redshifts to search
    :return: the indexes of the MgII doublets

    TODO move the calculation of matching reds outside of the trough loop so it is precomputed.
    """
    # data with wavelengths near MgII emission
    possible_troughs = passed_data['Observed Wavelength'][passed_data['Rest Wavelength'] > 2000]
    # cutoff trough indexes to minimum possible_mg2
    trough_indexes = trough_indexes[trough_indexes >= possible_troughs.index[0]]
    possible_troughs = possible_troughs.loc[trough_indexes]
    possible_troughs = possible_troughs.reset_index()

    # create dictionary of possible systems
    tagged_doublets = {}
    potential_matches = {}
    for z in redshifts:
        """
        Same as in determine_possible_redshifts(), right now we are only checking MgII so its value is 
        left in place and not looped for the time being.
        """
        potential_matches[z] = (float(doublets['Blue'].iloc[0] * (z + 1)),
                                float(doublets['Red'].iloc[0] * (z + 1)))

    for i in range(len(possible_troughs)):
        # unpack to get the separate graph index (x-value) and observed wavelength
        index = possible_troughs['index'].iloc[i]
        observed_wavelength = possible_troughs['Observed Wavelength'].iloc[i]
        redder_troughs = possible_troughs[possible_troughs['index'] > index]

        # if blue is close to potential blue and red is close to potential red, add the pair as a tuple to
        for z in redshifts:
            if np.isclose(observed_wavelength, potential_matches[z][0], atol=0.5):
                for j in range(len(redder_troughs)):
                    red_index = redder_troughs['index'].iloc[j]
                    red_observed_wavelength = redder_troughs['Observed Wavelength'].iloc[j]
                    if np.isclose(red_observed_wavelength, potential_matches[z][1], atol=0.5):
                        tagged_doublets[z] = (index, red_index)

    # remove redshifts that have no doublets
    for z in list(tagged_doublets):
        # if the tagged doublet is empty, equivalent to if tagged_doublets[z] == []
        if not tagged_doublets[z]:
            del tagged_doublets[z]

    return tagged_doublets


def theoretical_doublets(passed_data, confirmed_z, doublet):
    """
    Finds where we theoretically would expect to find the other doublets given the redshift of the MgII doublets.

    :param: passed_data the spectrum dataset
    :param: tagged_mgii_z the identified MgII doublets
    :return: the theoretical locations of other doublets corresponding to the redshifts with extant MgII doublet
    """
    redshifts = list(confirmed_z)
    theoretical = {}

    search_range = None

    match doublet:
        case "MgII":
            search_range = range(1, len(doublets))
        case "CIV":
            search_range = [i for i in range(len(doublets)) if i != 3]

    for z in redshifts:
        theoretical[z] = []

        # start at 1 so it doesn't re-do MgII
        for i in search_range:
            temp_doublet_name = doublets['Doublet'].iloc[i]
            temp_blue = float(doublets['Blue'].iloc[i] * (z + 1))
            temp_red = float(doublets['Red'].iloc[i] * (z + 1))

            # find the closest data index for plotting
            temp_blue_index = passed_data.index[(passed_data['Observed Wavelength'] - temp_blue).abs().idxmin()]
            temp_red_index = passed_data.index[(passed_data['Observed Wavelength'] - temp_red).abs().idxmin()]

            # only plot if the doublet could possibly appear on the spectrum
            if temp_red_index > 0:
                theoretical[z].append((temp_blue_index, temp_red_index, temp_doublet_name))

    return theoretical


def match_civ(passed_data, trough_indexes, redshifts):
    """
    Same as match_mgii but for carbon.
    The concept for this function is as follows: take each possible redshift generated by assuming a trough is a blue
    trough in a doublet. Then calculate the theoretical wavelength of the matching red trough and compare for each
    trough and redshift. If a checked trough is the same as the theoretical red trough, put the blue and red trough into
    a tuple and assign it to that redshift in a dictionary.

    :param: passed_data the spectrum dataset
    :param: troughs
    :param: redshifts the possible redshifts to search
    :return: the indexes of the MgII doublets

    TODO move the calculation of matching reds outside of the trough loop so it is precomputed.
    """
    # data with wavelengths near MgII emission
    possible_troughs = passed_data['Observed Wavelength'][passed_data['Rest Wavelength'] > 1000]
    # cutoff trough indexes to minimum possible_mg2
    trough_indexes = trough_indexes[trough_indexes >= possible_troughs.index[0]]
    possible_troughs = possible_troughs.loc[trough_indexes]
    possible_troughs = possible_troughs.reset_index()

    # create dictionary of possible systems
    tagged_doublets = {}
    potential_matches = {}
    for z in redshifts:
        """
        Same as in determine_possible_redshifts(), right now we are only checking MgII so its value is 
        left in place and not looped for the time being.
        """
        potential_matches[z] = (float(doublets['Blue'].iloc[3] * (z + 1)),
                                float(doublets['Red'].iloc[3] * (z + 1)))

    for i in range(len(possible_troughs)):
        # unpack to get the separate graph index (x-value) and observed wavelength
        index = possible_troughs['index'].iloc[i]
        observed_wavelength = possible_troughs['Observed Wavelength'].iloc[i]
        redder_troughs = possible_troughs[possible_troughs['index'] > index]

        # if blue is close to potential blue and red is close to potential red, add the pair as a tuple to
        for z in redshifts:
            if np.isclose(observed_wavelength, potential_matches[z][0], atol=0.5):
                for j in range(len(redder_troughs)):
                    red_index = redder_troughs['index'].iloc[j]
                    red_observed_wavelength = redder_troughs['Observed Wavelength'].iloc[j]
                    if np.isclose(red_observed_wavelength, potential_matches[z][1], atol=0.5):
                        tagged_doublets[z] = (index, red_index)

    # remove redshifts that have no doublets
    for z in list(tagged_doublets):
        # if the tagged doublet is empty, equivalent to if tagged_doublets[z] == []
        if not tagged_doublets[z]:
            del tagged_doublets[z]

    return tagged_doublets
