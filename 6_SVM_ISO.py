#  import libraries for loading data
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import label_binarize

#  import libraries for dimensionality reduction
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

x = pd.read_pickle(r"x_8000.pkl").values
y = pd.read_pickle(r"y_8000.pkl").values

#-------------------------------------------
# RUN CLASSIFIER WITH ISOMAP IMPLEMENTATION
#-------------------------------------------
classifier_condition = "SVM, ISOMAP"

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)

z=0
for train_index, test_index in cv.split(x,y):
    y_b = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_b.shape[1]

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_b[train_index], y_b[test_index] 
    
    start = time.time()
    
    embedding = Isomap(n_neighbors=5)
    x_train = embedding.fit_transform(x_train)
    x_test = embedding.fit_transform(x_test)

    #create SVM model
    svclassifier = svm.SVC(kernel='poly', random_state=5, gamma='scale', max_iter=5000, shrinking=False)
    classifier = OneVsRestClassifier(svclassifier)
    prediction = classifier.fit(x_train, y_train).decision_function(x_test)

    end = time.time()
    save_data[f"{classifier_condition}_fold_{z+1}_n=5"] = (model_evaluation(f"{classifier_condition}, 5", f"fold_{z+1}", x_test, y_test, prediction, classifier, end-start, n_classes))
    z+=1

save_data[f"Average {classifier_condition}, n = 5"] = save_data.iloc[:,11:].mean(axis=1)

save_data.to_csv(f"{classifier_condition}_8000_5.csv")