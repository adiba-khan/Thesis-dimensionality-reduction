from __future__ import absolute_import, division, print_function

#  import libraries for loading data
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import label_binarize

#  import libraries for classifier
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

#  import libraries for evaluation
from pprint import pprint
import time
import csv

def create_model():
    model = Sequential()
    model.add(Dense(x.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

x = pd.read_pickle(r"C:\Users\Adiba\Documents\Thesis\x.pkl").values
y = pd.read_pickle(r"C:\Users\Adiba\Documents\Thesis\y.pkl").values

#  binarize labels for multilabel auc calculations
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, verbose=0)

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
print("one")
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
print("hello")
grid_result = grid.fit(x, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

