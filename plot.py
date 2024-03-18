import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import trough_identify

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
    file = '106_14_41_51_14_20_50..txt'
    return file


def parse_shooter_number():
    """
    Gets the name of the object from file name
    """
    file = set_object_file()
    return file[:-5]


def pick_color():
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['magenta', 'green', 'brown', 'orange', 'darkred', 'goldenrod'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = enumerate(colors)
    return colors


def plot_systems(colors, ax, data, matched_mgii=None, matched_civ=None, theoretical_locations=None):
    """
    :param colors: the plot colors enumerator
    :param matched_mgii: MgII doublets
    :param ax: the plot axes
    :param data: the quasar data
    :param matched_civ: CIV doublets
    :param theoretical_locations: theoretical extrapolated locations of other doublets
    :return:
    """
    if matched_mgii:
        for mgii_z in matched_mgii:
            color = next(colors)[1]

            ax.vlines(data['Observed Wavelength'].loc[matched_mgii[mgii_z][0]], 0, data['Flux'].max(), color=color,
                      linewidth=0.6, label=str(np.round(mgii_z, decimals=3)) + " MgII")
            # plot red
            ax.vlines(data['Observed Wavelength'].loc[matched_mgii[mgii_z][1]], 0, data['Flux'].max(), color=color,
                      linewidth=0.6)

            # annotate
            ax.annotate(str(np.round(mgii_z, decimals=3)) + "MgII",
                        xy=(data['Observed Wavelength'].loc[matched_mgii[mgii_z][0]], 18),
                        xytext=(data['Observed Wavelength'].loc[matched_mgii[mgii_z][1]] + 5, 18),
                        rotation=270)

            if theoretical_locations:
                for theoretical_doublet in theoretical_locations[mgii_z]:
                    ax.vlines(data['Observed Wavelength'].loc[theoretical_doublet[0]], 0, data['Flux'].max(),
                              color=color, linewidth=0.6,
                              )
                    # plot red
                    ax.vlines(data['Observed Wavelength'].loc[theoretical_doublet[1]], 0, data['Flux'].max(),
                              color=color,
                              linewidth=0.6)

                    ax.annotate(str(np.round(mgii_z, decimals=3)) + " " + theoretical_doublet[2] + " Unconfirmed",
                                xy=(data['Observed Wavelength'].loc[theoretical_doublet[1]], 18),
                                xytext=(data['Observed Wavelength'].loc[theoretical_doublet[1]] + 5, 18),
                                rotation=270)
    if matched_civ:
        for civ_z in matched_civ:
            color = next(colors)[1]

            ax.vlines(data['Observed Wavelength'].loc[matched_civ[civ_z][0]], 0, data['Flux'].max(), color=color,
                      linewidth=0.6, label=str(np.round(civ_z, decimals=3)) + " CIV")
            # plot red
            ax.vlines(data['Observed Wavelength'].loc[matched_civ[civ_z][1]], 0, data['Flux'].max(), color=color,
                      linewidth=0.6)

            # annotate
            ax.annotate(str(np.round(civ_z, decimals=3)) + " CIV",
                        xy=(data['Observed Wavelength'].loc[matched_civ[civ_z][0]], 18),
                        xytext=(data['Observed Wavelength'].loc[matched_civ[civ_z][1]] + 5, 25),
                        rotation=270)

            if theoretical_locations:
                for theoretical_doublet in theoretical_locations[civ_z]:
                    ax.vlines(data['Observed Wavelength'].loc[theoretical_doublet[0]], 0, data['Flux'].max(),
                              color=color, linewidth=0.6,
                              )
                    # plot red
                    ax.vlines(data['Observed Wavelength'].loc[theoretical_doublet[1]], 0, data['Flux'].max(),
                              color=color,
                              linewidth=0.6)

                    ax.annotate(str(np.round(civ_z, decimals=3)) + " " + theoretical_doublet[2] + " Unconfirmed",
                                xy=(data['Observed Wavelength'].loc[theoretical_doublet[1]], 18),
                                xytext=(data['Observed Wavelength'].loc[theoretical_doublet[1]] + 5, 18),
                                rotation=270)


def plot_object():
    """
    Creates plot from text data file.
    """
    colors = pick_color()
    data = pd.read_csv('106_14_41_51_14_20_50..txt', header=None, delim_whitespace=True)
    data.columns = ('Observed Wavelength', 'Flux', 'Flux Error')
    # get all redshifts
    all_redshifts = pd.read_csv('Object_Index.txt', header=None)

    # get specific redshift
    object_num = parse_shooter_number()

    # From all_redshifts pulls column 3, row corresponding to object num, all characters except ending parenthesis.
    # Converts to int.
    z = float(all_redshifts[3][(int(object_num[:3]) - 100)][:-1])

    # create and insert column for rest wavelength
    rest_wavelength = data['Observed Wavelength'].map(lambda x: x / (1 + z))
    data['Rest Wavelength'] = rest_wavelength

    # find troughs
    troughs = trough_identify.identify_troughs(data, 1.3, z)

    possible_intervening_redshifts = trough_identify.determine_possible_redshifts(data, troughs[0])
    matched_mgii = trough_identify.match_mgii(data, troughs[0], possible_intervening_redshifts)
    matched_civ = trough_identify.match_civ(data, troughs[0], possible_intervening_redshifts)
    theoretical_locations_mgii = trough_identify.theoretical_doublets(data, matched_mgii, "MgII")
    theoretical_locations_civ = trough_identify.theoretical_doublets(data, matched_civ, "CIV")

    # create figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)
    plt.tight_layout()

    # plot flux and error
    ax.step(data['Observed Wavelength'], data['Flux'], linewidth=0.5, where="mid", color='#1f77b4')
    ax.step(data['Observed Wavelength'], data['Flux Error'], linewidth=1, color='red', where="mid")

    # rest wave axis on top
    rest_axis = ax.secondary_xaxis('top', functions=(lambda x: x / (1 + z), lambda x: x / (1 + z)))
    rest_axis.set_xlabel('Rest Wavelength')

    # plot troughs using indexes generated by identify_absorption applied to the observed wavelengths
    # ax.vlines(data['Observed Wavelength'].iloc[troughs[0]], 0, data['Flux'].max(), color='purple')

    # plot matched systems
    plot_systems(colors, ax, data, matched_mgii=matched_mgii, theoretical_locations=theoretical_locations_mgii)
    plot_systems(colors, ax, data, matched_civ=matched_civ, theoretical_locations=theoretical_locations_civ)

    # set plot limits
    ax.set_xlim(data['Observed Wavelength'].min(), data['Observed Wavelength'].max())
    ax.set_ylim(data['Flux'].min() - 1, data['Flux'].max() + 1)

    plt.legend()
    plt.show()


plot_object()
