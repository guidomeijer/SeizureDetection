# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:17:41 2023 by Guido Meijer
"""

import pywt
import numpy as np
import pandas as pd
from os.path import join
from skimage.transform import resize
from seizure_functions import paths, rms, bin_data

# Settings
N_SCALES = 64
WAVELET_DICT = dict({'acc_x': 'mexh', 'acc_y': 'mexh', 'acc_z': 'mexh', 'acc_mag': 'mexh',
                     'bvp': 'morl', 'eda': 'mexh', 'hr': 'mexh', 'temp': 'mexh'})

# Get paths
path_dict = paths()

# Load in training labels
train_labels = pd.read_csv(join(path_dict['data_path'], 'train', 'train_labels.csv'))

# Get random labels for negative class such that we have 5 consecutive data snippets 
neg_ind = np.random.choice(train_labels[train_labels['label'] == 0].index,
                           size=int(np.sum(train_labels['label'] == 1) / 5),
                           replace=False)
neg_ind = np.concatenate([np.arange(i, i+5) for i in neg_ind])
pos_ind = train_labels[train_labels['label'] == 1].index  # index to positive data
train_ind = np.concatenate((pos_ind, neg_ind))
train_y = np.concatenate((np.ones(pos_ind.shape), np.zeros(neg_ind.shape)))


train_data = np.ndarray(shape=(train_ind.shape[0], N_SCALES, N_SCALES, len(WAVELET_DICT)),
                        dtype='float32')
for i, filepath in enumerate(train_labels.loc[train_ind, 'filepath']):
    if np.mod(i, 100) == 0:
        print(f'Data snippet {i} of {len(train_ind)}')
    
    # Load in data snippet
    data_snippet = pd.read_parquet(join(path_dict['data_path'], 'train', filepath),
                                   engine='pyarrow')
    
    # Downsample data
    binned_data = bin_data(data_snippet, binsize=256, overlap=0.25)
    
    # Pre-allocate array
    X_cwt = np.ndarray(shape=(binned_data.shape[0], N_SCALES, N_SCALES, binned_data.shape[1]),
                       dtype='float32')
    
    for j, var in enumerate(binned_data.columns[:-1]):
        coeffs, freqs = pywt.cwt(binned_data['hr'], np.arange(1, N_SCALES+1),
                                 wavelet=WAVELET_DICT[var])
        rescale_coeffs = resize(coeffs, (N_SCALES, N_SCALES), mode='constant')
        train_data[i, :, :, j] = rescale_coeffs
        
# Save result
np.save(join(path_dict['data_path'], 'preprocessed_data', 'data_wavelet.npy'), train_data)
np.save(join(path_dict['data_path'], 'preprocessed_data', 'data_y.npy'), train_y)
