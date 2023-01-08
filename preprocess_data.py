# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:17:41 2023 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from seizure_functions import paths, rms

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

# Positive training data
train_data = pd.DataFrame()
for i, filepath in enumerate(train_labels.loc[pos_ind, 'filepath']):
    if np.mod(i, 100) == 0:
        print(f'Data snippet {i} of {len(pos_ind)}')
    
    # Load in data snippet
    data_snippet = pd.read_parquet(join(path_dict['data_path'], 'train', filepath))

    # Extract summary data
    train_data['temp_change'] = data_snippet['temp'].max() - data_snippet['temp'].min()
    train_data['hr_change'] = data_snippet['hr'].max() - data_snippet['hr'].min()
    train_data['acc_rms'] = rms(data_snippet['acc_mag']) 
    train_data['bvp_rms'] = rms(data_snippet['bvp']) 
    train_data['label'] = 1
    
# Negative training data
for i, filepath in enumerate(train_labels.loc[neg_ind, 'filepath']):
    if np.mod(i, 100) == 0:
        print(f'Data snippet {i} of {len(neg_ind)}')
    
    # Load in data snippet
    data_snippet = pd.read_parquet(join(path_dict['data_path'], 'train', filepath))

    # Extract summary data
    train_data['temp_change'] = data_snippet['temp'].max() - data_snippet['temp'].min()
    train_data['hr_change'] = data_snippet['hr'].max() - data_snippet['hr'].min()
    train_data['acc_rms'] = rms(data_snippet['acc_mag']) 
    train_data['bvp_rms'] = rms(data_snippet['bvp']) 
    train_data['label'] = 1