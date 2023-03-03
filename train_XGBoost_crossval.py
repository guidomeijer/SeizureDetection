# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:51:22 2023

@author: Guido
"""

import numpy as np
from seizure_functions import paths
from os.path import join
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Get paths
path_dict = paths()


def build_and_fit_xgb_model(X_train, y_train, X_test, y_test, n_depth, subsample, n_estimators):
    xgb_model = xgb.XGBClassifier(max_depth=n_depth, 
                                  objective='multi:softmax', # error evaluation for multiclass training
                                  num_class=2, 
                                  subsample=subsample, # randomly selected fraction of training samples that will be used to train each tree.
                                  n_estimators=n_estimators,
                                  eval_metric=['merror']) 
    eval_set = [(X_train, y_train), (X_test, y_test)]
    history = xgb_model.fit(X_train, y_train, eval_set=eval_set,verbose=True)
    return xgb_model, history


# Load in data
train_data = np.load(join(path_dict['data_path'], 'preprocessed_data', 'data_wavelet_PCA.npy'))
y = np.load(join(path_dict['data_path'], 'preprocessed_data', 'data_y_PCA.npy'))


# Select 90% to train and 10% to test
X_train, X_test, y_train, y_test = train_test_split(train_data, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

# define and train model    
trained_xgb_model, xgb_history = build_and_fit_xgb_model(X_train, y_train, X_test, y_test,
                                                         5, 0.5, 300)

# make predictions for test data
y_pred = trained_xgb_model.predict(X_test)
# determine the total accuracy 
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

y_prob = trained_xgb_model.predict_proba(X_test)
auroc = metrics.roc_auc_score(y_test, y_prob[:, 1])
print("Area under ROC curve: %.2f" % (auroc))