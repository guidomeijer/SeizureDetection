#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:37:42 2023

@author: guido
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pywt
import xgboost as xgb
from sklearn.decomposition import PCA

# Initialize
XBG_MODEL = "seizure_detection_xgb.model"
N_SCALES = 64
WAVELET_DICT = dict(
    {
        "acc_x": "mexh",
        "acc_y": "mexh",
        "acc_z": "mexh",
        "acc_mag": "mexh",
        "bvp": "morl",
        "eda": "mexh",
        "hr": "mexh",
        "temp": "mexh",
    }
)
pca = PCA(n_components=1)


def bin_data(data: pd.DataFrame, binsize=128) -> pd.DataFrame:
    
    binned_data = pd.DataFrame()
    for i, variable in enumerate(data.columns[1:]):
        this_array = np.array(data[variable])
        end = binsize * int(this_array.shape[0]/binsize)
        binned_data[variable] = np.mean(this_array[:end].reshape(-1, binsize), 1)
    return binned_data


def predict_seizure(data_snippet: pd.DataFrame) -> float:
    # Downsample data
    binned_data = bin_data(data_snippet, binsize=128)

    # Preprocess data
    this_X = []
    for j, var in enumerate(binned_data.columns[:-1]):
        
        # If this predictor is completely missing, fill with NaNs
        if np.sum(np.isnan(binned_data[var])) == binned_data[var].shape[0]:
            this_X.append([np.nan] * N_SCALES)
            continue
        
        # Do wavelet transform
        coeffs, freqs = pywt.cwt(
            binned_data[var][~np.isnan(binned_data[var])], np.arange(1, N_SCALES + 1), WAVELET_DICT[var]
        )
        
        # Do PCA on transform
        pca_transform = pca.fit_transform(coeffs).flatten()
        
        # Add to list
        this_X.append(pca_transform)

    # Add time of day (in hours)
    this_X.append([datetime.fromtimestamp(data_snippet['utc_timestamp'][0]).hour
                   + (datetime.fromtimestamp(data_snippet['utc_timestamp'][0]).minute / 60)])
        
    # Make list into array
    X = np.concatenate(this_X)
    
    # Load in model
    trained_xgb_model = xgb.XGBClassifier()  
    trained_xgb_model.load_model(XBG_MODEL)  

    # Predict seizure
    prob = trained_xgb_model.predict_proba([X])[0][1]

    return prob
