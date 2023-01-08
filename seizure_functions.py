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


def bin_data(data, binsize=256, overlap=0.5, method=np.mean):
    """
    binsize is in seconds
    """
    bin_centers = np.arange(binsize/2, data.shape[0], binsize).astype(int)
    binsize_act = binsize + int(binsize * overlap)  # actual binsize including overlap
    
    binned_data = pd.DataFrame()
    for i, variable in enumerate(data.columns[1:]):
        these_data = np.empty(bin_centers.shape)
        for b, bin_center in enumerate(bin_centers):
            bin_start = bin_center - int(binsize_act/2)
            bin_end = bin_center + int(binsize_act/2)
            if bin_start < 0:
                bin_start = 0
            if bin_end > data.shape[0]:
                bin_end = data.shape[0]
            these_data[b] = method(data[variable][bin_start:bin_end])
        binned_data[variable] = these_data
    binned_data['bin_centers'] = bin_centers
    return binned_data


def rms(y):
    return np.sqrt(np.mean(y**2))

if __name__ == "__main__":
    pathss = paths()
    print(pathss)
