#  import libraries for loading data
import pandas as pd
import numpy as np
import math 
from sklearn.preprocessing import label_binarize

#  import libraries for dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
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

#  load data
x = pd.read_pickle(r"x.pkl").values
y = pd.read_pickle(r"y.pkl").values

#-------------------------------------------------
# RUN CLASSIFIER WITH NO DR TECHNIQUE IMPLEMENTED
#-------------------------------------------------
classifier_condition = "Support Vector Machines (no DR technique)"

#  create model, evaluate, save data
#  z - counter to keep track of k-folds cross validation
z=0
print("hello")
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(x,y):
    y_b = label_binarize(y, classes=[0, 1, 2])
    print(z)
    n_classes = y_b.shape[1]
    # create train and test datasets
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_b[train_index], y_b[test_index]
    print("train and test created")    
    start = time.time()
    print("create model")
    svclassifier = svm.SVC(kernel='poly', random_state=5, gamma='scale', max_iter=5000, shrinking=False)
    print("run clssifier")
    classifier = OneVsRestClassifier(svclassifier, n_jobs=-1)
    print("create prediction")
    prediction = classifier.fit(x_train, y_train).decision_function(x_test)

    end = time.time()
    print("save data")
    save_data[f"{classifier_condition}_fold_{z+1}"] = (model_evaluation("SVM", f"fold_{z+1}", x_test, y_test, prediction, classifier, end-start, n_classes))
    print("data saved")
    z+=1

#  add column of averages of k-folds cross validation
save_data[f"Average {classifier_condition}"] = save_data.mean(axis=1)
#  save all SVM data
save_data.to_csv(f"{classifier_condition}.csv")

#----------------------------------------
# RUN CLASSIFIER WITH PCA IMPLEMENTATION
#----------------------------------------
#  create an empty dataframe to save evaluation metrics
data_columns = {'Data': ["execution time", "AUC label 0", "AUC label 1", "AUC label 2", "Overall Accuracy", "Precision-Recall label 0", "Precision-Recall label 1", "Precision-Recall label 2", "Average Precision"]}
save_data = pd.DataFrame(data=data_columns)

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

save_data.to_csv(f"{classifier_condition}.csv")

#------------------------------------------
# RUN CLASSIFIER WITH LASSO IMPLEMENTATION
#------------------------------------------
#  create an empty dataframe to save evaluation metrics
data_columns = {'Data': ["execution time", "AUC label 0", "AUC label 1", "AUC label 2", "Overall Accuracy", "Precision-Recall label 0", "Precision-Recall label 1", "Precision-Recall label 2", "Average Precision"]}
save_data = pd.DataFrame(data=data_columns)

classifier_condition = "SVM, LASSO"
alpha = [0.0001, 0.001, 0.01]

cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)

for idx in range(len(alpha)):
    z=0
    for train_index, test_index in cv.split(x,y):
        y_b = label_binarize(y, classes=[0, 1, 2])
        n_classes = y_b.shape[1]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_b[train_index], y_b[test_index]

        start = time.time()
        
        lasso = Lasso(alpha=alpha[idx]).fit(x_train, y_train)
        model = SelectFromModel(lasso, prefit=True)
        x_train = model.transform(x_train)
        x_test = model.transform(x_test)

        #create SVM model
        svclassifier = svm.SVC(kernel='poly', random_state=5, gamma='scale', max_iter=5000, shrinking=False)
        classifier = OneVsRestClassifier(svclassifier)
        prediction = classifier.fit(x_train, y_train).decision_function(x_test)

        end = time.time()
        save_data[f"{classifier_condition}_fold_{z+1}_alpha={alpha[idx]}"] = (model_evaluation(f"{classifier_condition}, {alpha[idx]}", f"fold_{z+1}", x_test, y_test, prediction, classifier, end-start, n_classes))
        z+=1

    range = ((idx*10) + (idx + 1))
    save_data[f"Average {classifier_condition}, n = {alpha[idx]}"] = save_data.iloc[:,range:].mean(axis=1)

save_data.to_csv(f"{classifier_condition}_new.csv")


#-------------------------------------------
# RUN CLASSIFIER WITH ISOMAP IMPLEMENTATION
#-------------------------------------------
'''ISOMAP is so slow that the value of n_components is manually adjusted;
the process in fact did not successfully run on the full dataset and various subsets of data
were created to generate results demonstrating that ISOMAP is disadvanatageous'''

#  create an empty dataframe to save evaluation metrics
data_columns = {'Data': ["execution time", "AUC label 0", "AUC label 1", "AUC label 2", "Overall Accuracy", "Precision-Recall label 0", "Precision-Recall label 1", "Precision-Recall label 2", "Average Precision"]}
save_data = pd.DataFrame(data=data_columns)

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

save_data.to_csv(f"{classifier_condition}_5.csv")
