from __future__ import absolute_import, division, print_function

#  import libraries for loading data
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import label_binarize

#  import libraries for dimensionality reduction
from sklearn.manifold import Isomap

#  import libraries for classifier
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#  import libraries for evaluation
import time
from eval_metrics import model_evaluation

print(tf.__version__)

x = pd.read_pickle(r"C:\Users\Adiba\Documents\Thesis\x_16000.pkl").values
y = pd.read_pickle(r"C:\Users\Adiba\Documents\Thesis\y_16000.pkl").values

data_columns = {'Data': ["execution time", "AUC label 0", "AUC label 1", "AUC label 2", "Overall Accuracy", "Precision-Recall label 0", "Precision-Recall label 1", "Precision-Recall label 2", "Average Precision"]}
save_data = pd.DataFrame(data=data_columns)

def build_model():
    model = Sequential()
    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#  create model, evaluate, save data
classifier_condition = "NN, ISOMAP"
ISO_var = [5]

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)

for idx in range(len(ISO_var)):
    z = 0
    for train_index, test_index in cv.split(x,y):
        y_b = label_binarize(y, classes=[0, 1, 2])
        n_classes = y_b.shape[1]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_b[train_index], y_b[test_index]

        start = time.time()

        embedding = Isomap(n_neighbors=ISO_var[idx])
        x_train = embedding.fit_transform(x_train)
        x_test = embedding.fit_transform(x_test)

        #create NN model
        model = build_model()
        model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1)

        prediction = model.predict(x_test)

        end = time.time()
        save_data[f"{classifier_condition}_fold_{z+1}_n={ISO_var[idx]}"] = (model_evaluation(f"{classifier_condition}, {ISO_var[idx]}_16000", f"fold_{z+1}", x_test, y_test, prediction, model, end-start, n_classes))

        z+=1

    range = ((idx*10) + (idx + 1))
    save_data[f"Average {classifier_condition}, n = {ISO_var[idx]}_4000"] = save_data.iloc[:,range:].mean(axis=1)

save_data.to_csv(f"{classifier_condition}_4000.csv")