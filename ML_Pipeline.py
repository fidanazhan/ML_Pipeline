# 1. Importing the libraries
#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier 

import matplotlib.pyplot as plt
print('Libraries Imported')
print()

# Data Input
dfData = pd.read_csv('Input Your Data')

factor = pd.factorize(dfData[-1])
dfData.Content = factor[0]
definitions = factor[1]
print(dfData.Content.head())
print(definitions)
print()

#Extracting Features and Output
#Splitting the data into independent variable and dependent variable
X = dfData.iloc[:,:-1]
y = dfData.iloc[:,-1]
print('The independent features set: ')
print(X.head(3))
print('The dependent variable set: ')
print(y.head(3))

#Train-Test Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 21)

# Metric To Evaluate ML Model 
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows
def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

# Fitting Random Forest Classification to the Training set
def random_forest(n_est, criteria, ran_state, X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators = n_est, criterion = criteria, random_state = ran_state)
    classifier.fit(X_train, y_train)

    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    reversefactor = dict(zip(range(4),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    
    #Confusion Matrix for calculation
    cm = confusion_matrix(y_test, y_pred)
    
    #Accuracy and Recall Score
    print("Random Forest for Accuracy and Recall: ")
    print("label precision recall")
    for label in range(2):
        print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    print()
    print("precision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))
    print()
    
    
    # Making the Confusion Matrix
    print('Random Forest Classifier: ')
    print()
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
    print('-----------------------------------------------------------------------------------------------------------')

def sup_vec_class(f_kernel, nilai_C, X_train, X_test, y_train, y_test):
    classifier = SVC(kernel = f_kernel, C = nilai_C)
    classifier.fit(X_train, y_train) 
    
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    
    #Confusion Matrix for calculation
    cm = confusion_matrix(y_test, y_pred)
    
    #Accuracy and Recall Score
    print("Support Vector Classifier for Accuracy and Recall: ")
    print("label precision recall")
    for label in range(2):
        print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    print()
    print("precision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))
    print()
    
    # Making the Confusion Matrix
    print('Support Vector Classifier Classifier: ')
    print()
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
    print('-----------------------------------------------------------------------------------------------------------')

def K_Neigbour(nilai_neighbour, X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors = nilai_neighbour)
    classifier.fit(X_train, y_train) 
    
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    
    #Confusion Matrix for calculation
    cm = confusion_matrix(y_test, y_pred)
    
    #Accuracy and Recall Score
    print("K Nearest Neighbour for Accuracy and Recall: ")
    print("label precision recall")
    for label in range(2):
        print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    print()
    print("precision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))
    print()
    
    # Making the Confusion Matrix
    print('K Nearest Neighbour Classifier: ')
    print()
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
    print('-----------------------------------------------------------------------------------------------------------')

def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    classifier = naive_bayes.GaussianNB()
    classifier.fit(X_train, y_train) 
    
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    
    #Confusion Matrix for calculation
    cm = confusion_matrix(y_test, y_pred)
    
    #Accuracy and Recall Score
    print("Naive Bayes for Accuracy and Recall: ")
    print("label precision recall")
    for label in range(2):
        print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    print()
    print("precision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))
    print()
    
    # Making the Confusion Matrix
    print('Naive Bayes Classifier: ')
    print()
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
    print('-----------------------------------------------------------------------------------------------------------')

def decision_tree(max_depthh, X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(max_depth = max_depthh)
    classifier.fit(X_train, y_train) 
    
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    reversefactor = dict(zip(range(4),definitions))
    y_test1 = np.vectorize(reversefactor.get)(y_test)
    y_pred1 = np.vectorize(reversefactor.get)(y_pred)
    
    #Confusion Matrix for calculation
    cm = confusion_matrix(y_test1, y_pred1)
    
    #Accuracy and Recall Score
    print("Decision Tree for Accuracy and Recall: ")
    print("label precision recall")
    for label in range(3):
        print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    print()
    print("precision total: ", precision_macro_average(cm))
    print("recall total: ", recall_macro_average(cm))
    f1_1 = (2*(precision_macro_average(cm)*recall_macro_average(cm)))/((precision_macro_average(cm)+recall_macro_average(cm)))

    
    # Making the Confusion Matrix
    print('Decision Tree Classifier: ')
    print()
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
    print('-----------------------------------------------------------------------------------------------------------')

sup_vec_class('linear', 1, X_train, X_test, y_train, y_test)
random_forest(10, 'entropy', 42, X_train, X_test, y_train, y_test)
naive_bayes_classifier(X_train, X_test, y_train, y_test)
decision_tree(4, X_train, X_test, y_train, y_test)

# -------------------------------------------------------------------------------------------------------------------------------------------

from sklearn import linear_model

# Create an instance of the classifier
classifier = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
#==============================================================================
# Data 
#==============================================================================

X = df.iloc[:,150:-1];
y = df.iloc[:,-1];

#==============================================================================
# CV MSE before feature selection
#==============================================================================
classifier = RandomForestClassifier(n_estimators = 42, criterion = 'entropy')
score = cross_val_score(classifier, X, y, cv=10, scoring="accuracy")
print("CV Random Forest Classifier before feature selection: {:.2f}".format(np.mean(score)))

classifier1 = DecisionTreeClassifier(max_depth = 5)
score = cross_val_score(classifier1, X, y, cv=10, scoring="accuracy")
print("CV Decision Tree Classifier before feature selection: {:.2f}".format(np.mean(score)))

classifier2 = naive_bayes.GaussianNB()
score = cross_val_score(classifier2, X, y, cv=10, scoring="accuracy")
print("CV Naive Bayes before feature selection: {:.2f}".format(np.mean(score)))

classifier3 = KNeighborsClassifier(n_neighbors = 1)
score = cross_val_score(classifier3, X, y, cv=10, scoring="accuracy")
print("CV K-NearestNeighbour before feature selection: {:.2f}".format(np.mean(score)))

classifier4 = SVC(kernel = 'linear', C = 2)
score = cross_val_score(classifier4, X, y, cv=10, scoring="accuracy")
print("CV SVM before feature selection: {:.2f}".format(np.mean(score)))

classifier5 = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
score = cross_val_score(classifier5, X, y, cv=10, scoring="accuracy")
print("CV Logistic Regression before feature selection: {:.2f}".format(np.mean(score)))