# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:17:41 2023 by Guido Meijer
"""

import pywt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from os.path import join
from seizure_functions import paths, bin_data

# Settings
N_SCALES = 64
N_TIMES_NEG = 4
WAVELET_DICT = dict({'acc_x': 'mexh', 'acc_y': 'mexh', 'acc_z': 'mexh', 'acc_mag': 'mexh',
                     'bvp': 'morl', 'eda': 'mexh', 'hr': 'mexh', 'temp': 'mexh'})
pca = PCA(n_components = 1)

def pca_of_cwt_coeffs(X, n_scales, wavelet_name="morl"):
    # apply PCA for just a single component to get the most significant coefficient per scale
    pca = PCA(n_components = 1)
    # create range of scales
    scales = np.arange(1, n_scales + 1)
   
    pca_comps = np.empty((0, n_scales),dtype='float32')
    for sample in range(X.shape[0]):
        coeffs, freqs = pywt.cwt(X, scales, wavelet_name)    
        pca_comps = np.vstack([pca_comps, pca.fit_transform(coeffs).flatten()]) 
                            
    return pca_comps


# Get paths
path_dict = paths()

# Load in training labels
train_labels = pd.read_csv(join(path_dict['data_path'], 'train', 'train_labels.csv'))

# Get random labels for negative class such that we have 5 consecutive data snippets 
neg_ind = np.random.choice(train_labels[train_labels['label'] == 0].index,
                           size=int((np.sum(train_labels['label'] == 1) / 5) * N_TIMES_NEG),
                           replace=False)
neg_ind = np.concatenate([np.arange(i, i+5) for i in neg_ind])
pos_ind = train_labels[train_labels['label'] == 1].index  # index to positive data
train_ind = np.concatenate((pos_ind, neg_ind))
train_y = np.concatenate((np.ones(pos_ind.shape), np.zeros(neg_ind.shape)))

train_data = np.ndarray(shape=(train_ind.shape[0], (N_SCALES * len(WAVELET_DICT)) + 1),
                        dtype='float32')
for i, filepath in enumerate(train_labels.loc[train_ind, 'filepath']):
    if np.mod(i, 100) == 0:
        print(f'Data snippet {i} of {len(train_ind)}')
    
    # Load in data snippet
    data_snippet = pd.read_parquet(join(path_dict['data_path'], 'train', filepath),
                                   engine='pyarrow')
    
    # Downsample data
    binned_data = bin_data(data_snippet, binsize=256, overlap=0.25)
        
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
        
    # Make list into array
    train_data[i, :-1] = np.concatenate(this_X)
        
    # Add time of day (in hours)
    train_data[i, -1] = (datetime.fromtimestamp(data_snippet['utc_timestamp'][0]).hour
                         + (datetime.fromtimestamp(data_snippet['utc_timestamp'][0]).minute / 60))
               
# Save result
np.save(join(path_dict['data_path'], 'preprocessed_data', 'data_wavelet_PCA.npy'), train_data)
np.save(join(path_dict['data_path'], 'preprocessed_data', 'data_y_PCA.npy'), train_y)
