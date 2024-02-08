import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

'''
Functions for generating the plot from text data files.

Jack Kelley
'''

'''
FUNCTIONS ------------------------------------------------------------------------------------------------------------
'''

'''
Sets the file to be used. 
TODO - make this take an input so the GUI can accept a file input
'''
def set_object_file():
    file = '108_11_03_26_31_41_15..txt'
    return file

'''
Gets the name of the object from file name
'''


def parse_shooter_number():
    file = set_object_file()
    return file[:-5]


'''
Creates plot from text data file.
'''


def plot_object():
    data = pd.read_csv('108_11_03_26_31_41_15..txt', header=None, delim_whitespace=True)
    data.columns = ('Observed Wavelength', 'Flux', 'Flux Error')
    # get all redshifts
    all_redshifts = pd.read_csv('Object_Index.txt', header=None)

    # get specific redshift
    object_num = parse_shooter_number()

    # From all_redshifts pulls column 3, row corresponding to object num, all characters except ending ')'.
    # Converts to int.
    z = float(all_redshifts[3][(int(object_num[:3]) - 100)][:-1])

    # create and insert column for rest wavelength
    rest_wavelength = data['Observed Wavelength'].map(lambda x: x / (1 + z))
    data['Rest Wavelength'] = rest_wavelength

    # create axes and figure objects
    figure, axes = plt.subplots()

    axes.plot(data[0], data[1])
    axes.secondary_xaxis
