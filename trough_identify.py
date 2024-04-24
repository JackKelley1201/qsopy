from scipy import signal
import numpy as np
import pandas as pd
from itertools import chain

"""
Finds the troughs in the spectrogram for emission identification

Jack Kelley
"""

doublets = data = pd.read_csv("line_list_short.txt", header=None, delim_whitespace=True)
doublets = doublets.drop(2, axis=1)
doublets = doublets.drop(4, axis=1)
doublets.columns = ("Doublet", "Blue", "Red")
doublets = doublets.sort_values(by="Blue", ascending=False)
doublets = doublets.reset_index(drop=True)


def identify_troughs(passed_data, prominence, distance=10):
    """
    Flips the data and identifies the troughs. Because the dataframe no longer starts at 0, add the first index
    to shift the indexes of the returned values by the same amount.

    TODO: set the min distance to allow a next peak to be greater than the maximum distance between the farthest
    doublet, then do a loop on each peak to find if there is another one with the distance equal to a doublet distance.
    """

    flux_data = passed_data["Flux"].loc[passed_data["Rest Wavelength"] > 1216]
    index_shift = flux_data.index[0]  # starting index is the point where rest wavelength is 1216
    flux_data = -flux_data
    smoothed_flux = signal.savgol_filter(flux_data, 8, 4)
    min_indexes = signal.find_peaks(smoothed_flux, prominence=prominence, distance=distance)
    min_indexes = min_indexes[0] + index_shift

    return min_indexes, index_shift


def determine_possible_redshifts(passed_data, trough_indexes, doublet_number, quasar_redshift):
    """
    Pass magnesium troughs to start with
    :param: passed_data the spectrum dataset
    :param: troughs the troughs that will be used to determine redshifts
    :param: element this is the index of the type of doublet from the doublets dataframe,
     so when trying to find MgII redshifts use 0, CIV redshifts use 3, etc.
    :param quasar_redshift: the redshift of the quasar, needed to stop adding when the redshift is greater than
     the quasar redshift
    :return:
    """

    # Lyman cutoff
    possible = passed_data["Observed Wavelength"][passed_data["Rest Wavelength"] > 1000]
    # cutoff trough indexes to Lyman cutoff
    trough_indexes = trough_indexes[trough_indexes >= possible.index[0]]
    # remove all wavelengths that aren't a trough
    possible = possible.loc[trough_indexes]

    # make list of possible redshifts
    intervening_z = []
    for trough in possible:
        """
        This gets the redshifts assuming that each trough is the blue one. There is no need to get the red ones,
        as the red ones would just be duplicates in a different order.
        """
        temp_z = (trough / float(doublets["Blue"].iloc[doublet_number])) - 1

        # stop adding if quasar redshift is less than the current calculated redshift
        if temp_z >= quasar_redshift:
            break

        intervening_z.append(temp_z)

    return intervening_z


def theoretical_doublets(passed_data, confirmed_z, doublet_number):
    """
    Finds where we theoretically would expect to find the other doublets given the redshift of an extant system

    :param passed_data: the spectrum dataset
    :param confirmed_z: the redshift of the confirmed systems
    :param doublet_number: the number of the doublet in the doublets dataframe
    :return: the theoretical locations of other doublets corresponding to the redshift of an extant system

    TODO: Look at why adding the quasar redshift give one correct iron but not the other
    """
    theoretical = {}

    # search all doublets except for the base doublet
    # note: this is very ugly, but it is the best way to concatenate two lists because they are their own objects
    search_range = chain(range(0, doublet_number), range(doublet_number + 1, len(doublets)))

    theoretical[confirmed_z] = []

    for i in search_range:
        temp_doublet_name = doublets["Doublet"].iloc[i]
        temp_blue = float(doublets["Blue"].iloc[i] * (confirmed_z + 1))
        temp_red = float(doublets["Red"].iloc[i] * (confirmed_z + 1))

        # find the closest data index for plotting, does this by subtracting the expected and getting the min extant
        # index, equivalent to finding the closest matching index to the expected value
        temp_blue_index = (passed_data["Observed Wavelength"] - temp_blue).abs().idxmin()
        temp_red_index = (passed_data["Observed Wavelength"] - temp_red).abs().idxmin()

        """
        Old method:
        temp_blue_index = passed_data.index[(passed_data["Observed Wavelength"] - temp_blue).abs().idxmin()]
        temp_red_index = passed_data.index[(passed_data["Observed Wavelength"] - temp_red).abs().idxmin()]
        """

        # only plot if the doublet could possibly appear on the spectrum
        if temp_red_index > 0:
            theoretical[confirmed_z].append((temp_blue_index, temp_red_index, temp_doublet_name))

    return theoretical


def match_confirmed_systems(passed_data, confirmed_redshift, doublet_number, trough_indexes):
    """
    Takes a confirmed redshift and finds all other doublets with that redshift
    :param passed_data: the data that the observed wavelengths are pulled from
    :param confirmed_redshift: the redshift that is used to find other doublets
    :param doublet_number: the number of the doublet that was already found
    :param trough_indexes: the troughs to search
    :return: the other doublets in the passed system
    """
    tagged_doublets = []

    search_range = chain(range(0, doublet_number), range(doublet_number + 1, len(doublets)))

    # data with wavelengths not in the forest
    possible_troughs = passed_data["Observed Wavelength"][passed_data["Rest Wavelength"] > 100]
    # cutoff trough indexes to minimum
    trough_indexes = trough_indexes[trough_indexes >= possible_troughs.index[0]]
    possible_troughs = possible_troughs.loc[trough_indexes]
    possible_troughs = possible_troughs.reset_index()

    # for each doublet number in the search range, check if there are doublets using the set of troughs
    for searched_doublet in search_range:
        potential_matches = float(doublets["Blue"].iloc[searched_doublet]
                                  * (confirmed_redshift + 1)), float(doublets["Red"].iloc[searched_doublet]
                                                                     * (confirmed_redshift + 1))

        for i in range(len(trough_indexes)):
            # unpack to get the separate graph index (x-value) and observed wavelength
            # unpack to get the separate graph index (x-value) and observed wavelength
            blue_index = possible_troughs["index"].iloc[i]
            observed_wavelength = possible_troughs["Observed Wavelength"].iloc[i]
            # only look at troughs redder than the current blue one to save time
            redder_troughs = possible_troughs[possible_troughs["index"] > blue_index]

            # if blue is close to potential blue and red is close to potential red, add the pair as a tuple to
            if np.isclose(observed_wavelength, potential_matches[0], atol=2):
                for j in range(len(redder_troughs)):
                    red_index = redder_troughs["index"].iloc[j]
                    red_observed_wavelength = redder_troughs["Observed Wavelength"].iloc[j]
                    if np.isclose(red_observed_wavelength, potential_matches[1], atol=2):
                        tagged_doublets.append((blue_index, red_index, searched_doublet))

    return tagged_doublets


def match_doublets(passed_data, trough_indexes, doublet_number, already_found, z):
    """
    Matches all doublets.

    Calls the redshift calculator within the function to avoid creating a massive list combining each doublet.

    The concept for this function is as follows: take each possible redshift generated by assuming a trough is a blue
    trough in a doublet. Then calculate the theoretical wavelength of the matching red trough and compare for each
    trough and redshift. If a checked trough is the same as the theoretical red trough, put the blue and red trough into
    a tuple and assign it to that redshift in a dictionary.

    :param: passed_data the spectrum dataset
    :param: troughs
    :param: redshifts the possible redshifts to search
    :return: the indexes of the MgII doublets
    """
    redshifts = determine_possible_redshifts(passed_data, trough_indexes, doublet_number, z)

    # remove redshifts less than 0.7
    redshifts = [redshift for redshift in redshifts if redshift > 0.7]

    # removed already found redshifts
    already_found_keys = list(already_found)
    for key in already_found_keys:
        already_found_redshifts = list(already_found[key])
        for already_found_redshift in already_found_redshifts:
            for i in reversed(range(0, len(redshifts))):
                if np.isclose(redshifts[i], already_found_redshift, atol=0.1):
                    redshifts.pop(i)

    # data with wavelengths not in the forest
    possible_troughs = passed_data["Observed Wavelength"][passed_data["Rest Wavelength"] > 1350]
    # cutoff trough indexes to minimum
    trough_indexes = trough_indexes[trough_indexes >= possible_troughs.index[0]]
    possible_troughs = possible_troughs.loc[trough_indexes]
    possible_troughs = possible_troughs.reset_index()

    # create dictionary of possible systems
    tagged_doublets = {}
    potential_matches = {}
    # calculate where we would expect to see a doublet for each redshift
    for z in redshifts:
        potential_matches[z] = (
            float(doublets["Blue"].iloc[doublet_number] * (z + 1)),
            float(doublets["Red"].iloc[doublet_number] * (z + 1)),
        )

    # get each trough and compare to the expected, if they match attach them to that redshift
    for i in range(len(possible_troughs)):
        # unpack to get the separate graph index (x-value) and observed wavelength
        blue_index = possible_troughs["index"].iloc[i]
        observed_wavelength = possible_troughs["Observed Wavelength"].iloc[i]
        # only look at troughs redder than the current blue one to save time
        redder_troughs = possible_troughs[possible_troughs["index"] > blue_index]

        # if blue is close to potential blue and red is close to potential red, add the pair as a tuple to
        for z in redshifts:
            if np.isclose(observed_wavelength, potential_matches[z][0], atol=2.5):
                for j in range(len(redder_troughs)):
                    red_index = redder_troughs["index"].iloc[j]
                    red_observed_wavelength = redder_troughs["Observed Wavelength"].iloc[j]
                    if np.isclose(red_observed_wavelength, potential_matches[z][1], atol=2.5):
                        tagged_doublets[z] = [(blue_index, red_index)]

    # remove redshifts that have no doublets
    for z in list(tagged_doublets):
        # if the tagged doublet is empty, equivalent to if tagged_doublets[z] == []
        if not tagged_doublets[z]:
            del tagged_doublets[z]

    return tagged_doublets


def theoretical_quasar(passed_data, redshift):
    """
    Finds where we theoretically would expect to find the other doublets given the redshift of the MgII doublets.

    :param passed_data: the spectrum dataset
    :param redshift: the quasar redshift
    :return: the theoretical locations of other doublets corresponding to the redshift of the quasar
    """
    theoretical = {}

    # search all doublets except for the base doublet
    # note: this is very ugly, but it is the best way to concatenate two lists because they are their own objects
    search_range = range(0, len(doublets))

    theoretical[redshift] = []

    for i in search_range:
        temp_doublet_name = doublets["Doublet"].iloc[i]
        temp_blue = float(doublets["Blue"].iloc[i] * (redshift + 1))
        temp_red = float(doublets["Red"].iloc[i] * (redshift + 1))

        # find the closest data index for plotting, does this by subtracting the expected and getting the min extant
        # index, equivalent to finding the closest matching index to the expected value
        temp_blue_index = passed_data.index[(passed_data["Observed Wavelength"] - temp_blue).abs().idxmin()]
        temp_red_index = passed_data.index[(passed_data["Observed Wavelength"] - temp_red).abs().idxmin()]

        # only plot if the doublet could possibly appear on the spectrum
        if temp_red_index > 0:
            theoretical[redshift].append(
                (temp_blue_index, temp_red_index, temp_doublet_name)
            )

    return theoretical
