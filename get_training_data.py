# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:09:46 2022 by Guido Meijer
"""

from seizure_functions import paths
import pandas as pd

path_dict = paths()

test = pd.read_parquet('D:\\SeizureDetection\\train\\data\\parquet\\train\\1110\\011\\UTC-2020_03_08-11_00_00.parquet',
                       engine='pyarrow')

labels = pd.read_csv('D:\\SeizureDetection\\train\\train_labels.csv')