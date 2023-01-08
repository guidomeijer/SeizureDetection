# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:51:22 2023

@author: Guido
"""

import numpy as np
from seizure_functions import paths
from os.path import join
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
#import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# Get paths
path_dict = paths()


def build_cnn_model(activation, input_shape):
    model = Sequential()
    
    # 2 Convolution layer with Max polling
    model.add(Conv2D(32, 5, activation=activation, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
    model.add(MaxPooling2D())  
    model.add(Flatten())
    
    # 3 Full connected layer
    model.add(Dense(128, activation=activation, kernel_initializer="he_normal"))
    model.add(Dense(54, activation=activation, kernel_initializer="he_normal"))
    model.add(Dense(6, activation='softmax')) # 6 classes
    
    # summarize the model
    print(model.summary())
    return model


def compile_and_fit_model(model, X_train, y_train, X_test, y_test, batch_size, n_epochs):

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    
    # define callbacks
    callbacks = [ModelCheckpoint(filepath='best_model.h5',
                                 monitor='val_sparse_categorical_accuracy',
                                 save_best_only=True)]
    
    # fit the model
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))
    
    return model, history


# Load in data
train_data = np.load(join(path_dict['data_path'], 'preprocessed_data', 'data_wavelet.npy'))
y = np.load(join(path_dict['data_path'], 'preprocessed_data', 'data_y.npy'))

# Clean up data
for i in range(train_data.shape[0]):
    

# shape of the input images
input_shape = (X_train_cwt.shape[1], X_train_cwt.shape[2], X_train_cwt.shape[3])

# create cnn model
cnn_model = build_cnn_model("relu", input_shape)
# train cnn model
trained_cnn_model, cnn_history = compile_and_fit_model(cnn_model, X_train_cwt, y_train, X_test_cwt, y_test, 368, 10)