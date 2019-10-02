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

#  import libraries for classifier
from sklearn import svm, datasets, preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

#  import libraries for evaluation
import time
from eval_metrics import model_evaluation

#  create an empty dataframe to save evaluation metrics
data_columns = {'Data': ["execution time", "AUC label 0", "AUC label 1", "AUC label 2", "Overall Accuracy", "Precision-Recall label 0", "Precision-Recall label 1", "Precision-Recall label 2", "Average Precision"]}
save_data = pd.DataFrame(data=data_columns)

x = pd.read_pickle(r"x.pkl").values
y = pd.read_pickle(r"y.pkl").values

#----------------------------------------
# RUN CLASSIFIER WITH PCA IMPLEMENTATION
#----------------------------------------
classifier_condition = "SVM, PCA"
PCA_var = [0.95, 0.90, 0.85]

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)

for idx in range(len(PCA_var)):
    z=0
    for train_index, test_index in cv.split(x,y):

        y_b = label_binarize(y, classes=[0, 1, 2])
        n_classes = y_b.shape[1]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_b[train_index], y_b[test_index]

        start = time.time()

        pca = PCA(n_components=PCA_var[idx], svd_solver="full")
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

        #create SV Mmodel
        svclassifier = svm.SVC(kernel='poly', random_state=5, gamma='scale', max_iter=5000, shrinking=False)
        classifier = OneVsRestClassifier(svclassifier)
        prediction = classifier.fit(x_train, y_train).decision_function(x_test)

        end = time.time()
        save_data[f"{classifier_condition}_fold_{z+1}_n={PCA_var[idx]}"] = (model_evaluation(f"{classifier_condition}, {PCA_var[idx]}", f"fold_{z+1}", x_test, y_test, prediction, classifier, end-start, n_classes))
        z+=1

    range = ((idx*10) + (idx + 1))
    save_data[f"Average {classifier_condition}, n = {PCA_var[idx]}"] = save_data.iloc[:,range:].mean(axis=1)

save_data.to_csv(f"{classifier_condition}_new_PCA.csv")
