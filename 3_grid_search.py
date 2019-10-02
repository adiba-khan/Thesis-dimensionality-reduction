#cross validation https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
#  import libraries for loading data
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import label_binarize

#  import libraries for classifier
from sklearn import svm, datasets, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pickle

x = pd.read_pickle(r"x.pkl").values
y = pd.read_pickle(r"y.pkl").values

y = label_binarize(y, classes=[0, 1, 2])

def grid_search_results(estimator, grid_param, classifier):
    gd_sr = GridSearchCV(estimator=estimator, param_grid=grid_param,
        scoring='accuracy', cv=5, n_jobs=-1)

    gd_sr.fit(x, y)
    results = []
    results.append([gd_sr.best_params_, gd_sr.best_score_])
    print(results)

    with open(f'parrot_{classifier}.pkl', 'wb') as f:
        pickle.dump(x, f)

"""RF_grid_param = {'n_estimators': [100, 500, 1000],
    'criterion': ['gini', 'entropy']}

rfclassifier = RandomForestClassifier(random_state=5)

grid_search_results(rfclassifier, RF_grid_param, "Random_Forest")"""

SVM_grid_param = {'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 0.1, 1, 10],
    'shrinking': [True, False]}

svclassifier = svm.SVC(probability=False, random_state=5)

grid_search_results(svclassifier, SVM_grid_param, "SVM")