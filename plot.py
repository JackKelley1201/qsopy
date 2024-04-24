import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.transforms import Bbox
from scipy import signal
import os
import trough_identify

"""
FIELDS ---------------------------------------------------------------------------------------------------------------
"""

"""
Functions for generating the plot from text data files.

Jack Kelley
"""

"""
FUNCTIONS ------------------------------------------------------------------------------------------------------------
"""
# List of bounding boxes for annotations to help make sure none of them overlap
annotation_bounding_boxes = []


def set_object_file():
    """
    Sets the file to be used.
    TODO - make this take an input so the GUI can accept a file input
    """
    file = "106_14_41_51_14_20_50..txt"
    return file


def parse_shooter_number():
    """
    Gets the name of the object from file name
    """
    file = set_object_file()
    return file[:-5]


def pick_color():
    # color=plt.cm.viridis(np.linspace(0, 1, 25))
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 5)))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors = enumerate(colors)

    return colors


def plot_systems(colors, ax, data, doublet_number, all_matched_doublets):
    """
    :param colors: the plot colors enumerator
    :param ax: the plot axes
    :param data: the quasar data
    :param doublet_number: the number of the doublets in the doublets dataframe in trough_identify
    :param all_matched_doublets: the set of detected doublets
    :return:

    Note: the theoretical doublet structure is a dictionary even though the current implementation does not strictly
    require it. I did this in case later on we want to combine all the theoretical redshifts for something. That is
    why it may look sort of strange/redundant.
    """

    matched_doublets = all_matched_doublets[doublet_number]
    for matched_doublet in matched_doublets:

        main_doublet = matched_doublets[matched_doublet][0]
        secondary_doublets = matched_doublets[matched_doublet][1]

        color = next(colors)[1]

        # plot blue
        ax.vlines(
            data["Observed Wavelength"].loc[main_doublet[0]],
            0,
            data["Flux"].max(),
            color=color,
            linewidth=0.8,
            label=str(np.round(matched_doublet, decimals=3)) + trough_identify.doublets["Doublet"].iloc[doublet_number]
        )

        # plot red
        ax.vlines(
            data["Observed Wavelength"].loc[main_doublet[1]],
            0,
            data["Flux"].max(),
            color=color,
            linewidth=0.8
        )

        # annotate
        ax.annotate(
            str(np.round(matched_doublet, decimals=3))
            + trough_identify.doublets["Doublet"].loc[doublet_number],
            xy=(
                data["Observed Wavelength"].loc[main_doublet[0]],
                18,
            ),
            xytext=(
                data["Observed Wavelength"].loc[main_doublet[1]]
                + 5,
                18,
            ),
            rotation=270
        )

        for secondary in secondary_doublets:
            # plot line
            ax.vlines(
                data["Observed Wavelength"].loc[secondary[0]],
                0,
                data["Flux"].max(),
                color=color,
                linewidth=0.8
            )

            # annotate

            # offset the annotation if there is already an annotation in the same place
            temp_offset = 0
            temp_text = ""
            temp_bounding_box = Bbox.from_bounds(data["Observed Wavelength"].loc[secondary[0]],
                                                 18, 5, 5)
            if temp_bounding_box.count_overlaps(annotation_bounding_boxes) != 0:
                temp_offset += 5
                temp_text = "--OR-- "

            ax.annotate(
                temp_text
                + str(np.round(matched_doublet, decimals=3))
                + trough_identify.prediction_singlets["Doublet"].loc[secondary[1]]
                + " Predicted",
                xy=(
                    data["Observed Wavelength"].loc[secondary[0]],
                    18
                ),
                xytext=(
                    data["Observed Wavelength"].loc[secondary[0]]
                    + 5,
                    18 - temp_offset,
                ),
                rotation=270
            )
            annotation_bounding_boxes.append(temp_bounding_box)


def plot_quasar_system(ax, data, quasar_redshift):
    """
    :param ax: the plot axes
    :param data: the quasar data
    :param quasar_redshift: the given quasar redshift
    :return:
    """

    quasar_system = trough_identify.theoretical_quasar(data, quasar_redshift)
    color = "red"
    redshift = list(quasar_system)[0]

    # plot blue
    for doublet in quasar_system[redshift]:
        ax.vlines(
            data["Observed Wavelength"].loc[doublet[0]],
            0,
            data["Flux"].max(),
            color=color,
            linewidth=0.6
        )

        # plot red
        ax.vlines(
            data["Observed Wavelength"].loc[doublet[1]],
            0,
            data["Flux"].max(),
            color=color,
            linewidth=0.6
        )

        # annotate
        ax.annotate(
            str(np.round(redshift, decimals=3))
            + doublet[2],
            xy=(
                data["Observed Wavelength"].loc[doublet[0]],
                18
            ),
            xytext=(
                data["Observed Wavelength"].loc[doublet[1]]
                + 5,
                18
            ),
            rotation=270
        )


def plot_object():
    """
    Creates plot from text data file.
    """
    colors = pick_color()
    data = pd.read_csv("106_14_41_51_14_20_50..txt", header=None, delim_whitespace=True)
    data.columns = ("Observed Wavelength", "Flux", "Flux Error")
    # get all redshifts
    all_redshifts = pd.read_csv("Object_Index.txt", header=None)

    # get specific redshift
    object_num = parse_shooter_number()

    # From all_redshifts pulls column 3, row corresponding to object num, all characters except ending parenthesis.
    # Converts to int.
    z = float(all_redshifts[3][(int(object_num[:3]) - 100)][:-1])

    # create and insert column for rest wavelength
    rest_wavelength = data["Observed Wavelength"].map(lambda x: x / (1 + z))
    data["Rest Wavelength"] = rest_wavelength

    # find troughs
    troughs = trough_identify.identify_troughs(data, 2)
    higher_sensitivity_troughs = trough_identify.identify_troughs(data, 1.6, distance=2)

    all_matched_doublets = {}
    for i in range(len(trough_identify.doublets)):
        all_matched_doublets[i] = trough_identify.match_doublets(data, troughs[0], i, all_matched_doublets, z)

        for redshift in all_matched_doublets[i]:
            all_matched_doublets[i][redshift].append(
                trough_identify.match_confirmed_systems(data, redshift, i, higher_sensitivity_troughs[0]))

    # create figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)
    plt.tight_layout()

    # plot flux and error
    ax.step(
        data["Observed Wavelength"],
        data["Flux"],
        linewidth=0.5,
        where="mid",
        color="#1f77b4",
    )
    ax.step(
        data["Observed Wavelength"],
        data["Flux Error"],
        linewidth=1,
        color="red",
        where="mid",
    )

    # rest wave axis on top
    rest_axis = ax.secondary_xaxis("top", functions=(lambda x: x / (1 + z), lambda x: x / (1 + z)))
    rest_axis.set_xlabel("Rest Wavelength")

    # plot troughs using indexes generated by identify_absorption applied to the observed wavelengths
    # ax.vlines(data['Observed Wavelength'].iloc[troughs[0]], 0, data['Flux'].max(), color='purple')

    # plot matched systems
    for i in range(0, 4):
        plot_systems(colors, ax, data, i, all_matched_doublets)

    # plot_quasar_system(ax, data, z)

    # set plot limits
    ax.set_xlim(data["Observed Wavelength"].min(), data["Observed Wavelength"].max())
    ax.set_ylim(data["Flux"].min() - 1, data["Flux"].max() + 1)

    # plot smoothed data
    # plt.clf()
    # plt.step(data["Observed Wavelength"], signal.savgol_filter(data["Flux"], 8, 4))

    plt.legend()
    plt.show()


plot_object()
