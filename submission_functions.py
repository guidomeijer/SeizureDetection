#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:37:42 2023

@author: guido
"""

import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from sklearn.decomposition import PCA

# Initialize
N_SCALES = 64
WAVELET_DICT = dict({'acc_x': 'mexh', 'acc_y': 'mexh', 'acc_z': 'mexh', 'acc_mag': 'mexh',
                     'bvp': 'morl', 'eda': 'mexh', 'hr': 'mexh', 'temp': 'mexh'})
pca = PCA(n_components=1)


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


def predict_seizure(data_snippet):

    # Downsample data
    binned_data = bin_data(data_snippet, binsize=256, overlap=0.25)
       
    # Preprocess data
    this_X = []
    for j, var in enumerate(binned_data.columns[:-1]):
        coeffs, freqs = pywt.cwt(binned_data[var], np.arange(1, N_SCALES+1), WAVELET_DICT[var])
        this_X.append(pca.fit_transform(coeffs).flatten())
    X = np.concatenate(this_X)
    
    # Load in model
    trained_xgb_model = xgb.Booster({'nthread': 4})  # init model
    trained_xgb_model.load_model('seizure_detection_xgb.model')  # load data
    
    # Predict seizure
    prob = trained_xgb_model.predict_proba([X])[0][1]
    
    return prob
    
        
        
        