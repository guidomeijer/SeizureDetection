# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:51:22 2023

@author: Guido
"""

import numpy as np
from seizure_functions import paths
from os.path import join
import xgboost as xgb


# Get paths
path_dict = paths()


def build_and_fit_xgb_model(X, y, n_depth, subsample, n_estimators):
    xgb_model = xgb.XGBClassifier(max_depth=n_depth,
                                  objective='multi:softmax', # error evaluation for multiclass training
                                  num_class=2,
                                  subsample=subsample, # randomly selected fraction of training samples that will be used to train each tree.
                                  n_estimators=n_estimators)
    history = xgb_model.fit(X, y, verbose=True)
    return xgb_model, history


# Load in data
train_data = np.load(join(path_dict['data_path'], 'preprocessed_data', 'data_wavelet_PCA.npy'))
y = np.load(join(path_dict['data_path'], 'preprocessed_data', 'data_y_PCA.npy'))

# define and train model
trained_xgb_model, xgb_history = build_and_fit_xgb_model(train_data, y, 5, 0.5, 300)

# save
trained_xgb_model.save_model('seizure_detection_xgb.model')
