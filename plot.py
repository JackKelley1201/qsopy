import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from trough_identify import identify_absorption

"""
Functions for generating the plot from text data files.

Jack Kelley
"""

"""
FUNCTIONS ------------------------------------------------------------------------------------------------------------
"""


def set_object_file():
    """
    Sets the file to be used.
    TODO - make this take an input so the GUI can accept a file input
    """
    file = '108_11_03_26_31_41_15..txt'
    return file


def parse_shooter_number():
    """
    Gets the name of the object from file name
    """
    file = set_object_file()
    return file[:-5]


def plot_object():
    """
    Creates plot from text data file.
    """
    data = pd.read_csv('108_11_03_26_31_41_15..txt', header=None, delim_whitespace=True)
    data.columns = ('Observed Wavelength', 'Flux', 'Flux Error')
    # get all redshifts
    all_redshifts = pd.read_csv('Object_Index.txt', header=None)

    # get specific redshift
    object_num = parse_shooter_number()

    # find troughs
    troughs = identify_absorption(data['Flux'], 2)

    # From all_redshifts pulls column 3, row corresponding to object num, all characters except ending ')'.
    # Converts to int.
    z = float(all_redshifts[3][(int(object_num[:3]) - 100)][:-1])

    # create and insert column for rest wavelength
    rest_wavelength = data['Observed Wavelength'].map(lambda x: x / (1 + z))
    data['Rest Wavelength'] = rest_wavelength

    # create figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)
    plt.tight_layout()

    # plot flux and error
    ax.plot(data['Observed Wavelength'], data['Flux'], linewidth=0.5)
    ax.plot(data['Observed Wavelength'], data['Flux Error'], linewidth=1, color='red')

    # rest wave axis on top
    rest_axis = ax.secondary_xaxis('top', functions=(lambda x: x / (1 + z), lambda x: x / (1 + z)))
    rest_axis.set_xlabel('Rest Wavelength')

    # plot troughs using indexes generated by identify_absorption applied to the observed wavelengths
    ax.vlines(data['Observed Wavelength'].iloc[troughs[0]], 0, data['Flux'].max(), color='purple')

    # set plot limits
    ax.set_xlim(data['Observed Wavelength'].min(), data['Observed Wavelength'].max())
    ax.set_ylim(data['Flux'].min() - 1, data['Flux'].max() + 1)

    plt.legend()
    plt.show()


plot_object()
