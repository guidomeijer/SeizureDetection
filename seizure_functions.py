# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:05:19 2022 by Guido Meijer
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from typing import Dict
from os.path import join, dirname, realpath, isfile


def paths() -> Dict[str, str]:
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        paths = dict()
        paths['fig_path'] = input('Path folder to save figures: ')
        paths['data_path'] = input('Path to data folder:')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(paths, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        paths = json.load(json_file)
    paths['repo_path'] = dirname(realpath(__file__))
    return paths


def bin_data(data, binsize=60, method=np.mean):
    """
    binsize is in seconds
    """
    binned_data = pd.DataFrame()
    time_ax = np.array(data['utc_timestamp'] - data.loc[0, 'utc_timestamp'])
    bins = np.arange(0, time_ax[-1] + binsize, binsize)
    for i, variable in enumerate(data.columns[1:]):
        binned_data[variable] = binned_statistic(time_ax, data[variable],
                                                 bins=bins,
                                                 statistic=method)[0]
    binned_data['bin_centers'] = bins[:-1] + (np.diff(bins) / 2)
    return binned_data


def rms(y):
    return np.sqrt(np.mean(y**2))

if __name__ == "__main__":
    pathss = paths()
    print(pathss)
