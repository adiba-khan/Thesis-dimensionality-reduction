#  import libraries for loading data
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import label_binarize

#  import libraries for dimensionality reduction
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import Isomap

#  import libraries for classifier
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

#  import libraries for evaluation
import time
from eval_metrics import model_evaluation

data_columns = {'Data': ["execution time", "AUC label 0", "AUC label 1", "AUC label 2", "Overall Accuracy", "Precision-Recall label 0", "Precision-Recall label 1", "Precision-Recall label 2", "Average Precision"]}
save_data = pd.DataFrame(data=data_columns)

x = pd.read_pickle(r"x.pkl").values
y = pd.read_pickle(r"y.pkl").values

#  binarize labels for multilabel auc calculations
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

#---------------------------
# RUN CLASSIFIER WITH NO DR
#---------------------------
classifier_condition = "Random Forest, no DR"

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.20, random_state=5)

rfclassifier = RandomForestClassifier(n_estimators=500, random_state=5, criterion = 'gini')
classifier = OneVsRestClassifier(rfclassifier, n_jobs = -1)
classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

end = time.time()

save_data[f"{classifier_condition}_n = {PCA_var[idx]}"] = (model_evaluation("RF", PCA_var[idx], x_test, y_test, prediction, classifier, end-start, n_classes))

#----------------------------------------
# RUN CLASSIFIER WITH PCA IMPLEMENTATION
#----------------------------------------
classifier_condition = "Random Forest, PCA"
PCA_var = [0.95, 0.90, 0.85]

#(219, 207, 196)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.20, random_state=5)

for idx in range(len(PCA_var)):
    start = time.time()

    pca = PCA(n_components=PCA_var[idx], svd_solver="full")
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    rfclassifier = RandomForestClassifier(n_estimators=500, random_state=5, criterion = 'gini')
    classifier = OneVsRestClassifier(rfclassifier, n_jobs = -1)
    classifier.fit(x_train, y_train)

    prediction = classifier.predict(x_test)

    end = time.time()

    save_data[f"{classifier_condition}_n = {PCA_var[idx]}"] = (model_evaluation("RF", PCA_var[idx], x_test, y_test, prediction, classifier, end-start, n_classes))

#----------------------------------------
# RUN CLASSIFIER WITH LASSO IMPLEMENTATION
#----------------------------------------

classifier_condition = "Random Forest, LASSO"
alpha = [0.0001, 0.001, 0.01]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.20, random_state=5)

for idx in range(len(alpha)):
    start = time.time()

    pca = PCA(n_components=alpha[idx], svd_solver="full")
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    rfclassifier = RandomForestClassifier(n_estimators=500, random_state=5, criterion = 'gini')
    classifier = OneVsRestClassifier(rfclassifier, n_jobs = -1)
    classifier.fit(x_train, y_train)

    prediction = classifier.predict(x_test)

    end = time.time()

    save_data[f"{classifier_condition}_n = {PCA_var[idx]}"] = (model_evaluation("RF", alpha[idx], x_test, y_test, prediction, classifier, end-start, n_classes))

    save_data.to_csv("Random_Forest_Results.csv"

                     
#-------------------------------------------
# RUN CLASSIFIER WITH ISOMAP IMPLEMENTATION
#-------------------------------------------
'''ISOMAP is so slow that the value of n_components is manually adjusted;
the process in fact did not successfully run on the full dataset and various subsets of data
were created to generate results demonstrating that ISOMAP is disadvanatageous'''
                     
classifier_condition = "Random Forest, ISOMAP"

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.20, random_state=5)

start = time.time()

embedding = Isomap(n_components=6)
x_train = embedding.fit_transform(x_train)
x_test = embedding.fit_transform(x_test)

rfclassifier = RandomForestClassifier(n_estimators=500, random_state=5, criterion = 'gini')
classifier = OneVsRestClassifier(rfclassifier, n_jobs=-1)
classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

end = time.time()

save_data[f"{classifier_condition}_n = 6"] = (model_evaluation("RF", "6", x_test, y_test, prediction, classifier, end-start, n_classes))

save_data.to_csv("Random_Forest_ISO_6.csv")
