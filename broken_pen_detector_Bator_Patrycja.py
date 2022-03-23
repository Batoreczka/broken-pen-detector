# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:39:26 2022

@author: bator
"""
############# Machine_Learning_Project_Patrycja_Bator
#
# Problem: recognize if there is a good pen or broken pen in the photo;
#   in: photos of good and broken pens are divided into training and test pens;
#   out: prediction of photos from the test set;
#
# Algorithm:
# 1. Upload photos and prepare training data and testing X and y.
# 2. Test some ML algorithms.
# 3. Present the result.
#
#############

### 1. Data loading

import cv2
import os
import numpy as np
print(os.getcwd())

def read_dataset(path_good, path_bad):
    X = []
    y = []
    
    file_names = [(path_good + name, "good") for name in os.listdir(path_good)]
    file_names += [(path_bad + name, "bad") for name in os.listdir(path_bad)]
    
    for name, decision in file_names:
        X.append(read_img(name))
        y.append(decision)
    
    return X, y

def read_img(path):
    img = cv2.imread(path)
    
    histogram = cv2.calcHist([img],[0,1,2],None,[8,8,8], [0,255,0,255,0,255]) # computing the histogram
    cv2.normalize(histogram, histogram) # normalization of the histogram
    histogram = histogram.flatten()
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # shifting to grayscale
    img = cv2.erode(img, np.ones((5,5), np.uint8), iterations=3) # sharpening
    img = cv2.resize(img, (8,8)) ### size changed to 8x8
    img_moments = cv2.HuMoments(cv2.moments(img)).flatten()
    img_vector = np.hstack([img_moments, histogram]) # concatenation moment and histogram
    
    return img_vector 
 
    # img = cv2.dilate(img, np.ones((5,5), np.uint8), iterations=2) ### dylatyzacja
    # cv2.imwrite("gray_scale/test_good_01_dilate.jpg",img)

X_tr, y_tr = read_dataset("training set/good/", "training set/bad/")
X_te, y_te = read_dataset("test set/good/", "test set/bad/")

### 2. Machine Learning
### classifiers

classifiers = {} # dictionary of classifiers

from sklearn import tree # CART Decision Tree
classifiers["DecisionTree"] = tree.DecisionTreeClassifier(max_depth=2)

from sklearn import svm 
classifiers["SVM"] = svm.SVC(kernel="linear", C=1) # linear SVM classifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

classifiers["RandomForest"] = RandomForestClassifier(max_depth=2, n_estimators=100) # a random decision forest classifier

classifiers["AdaBoost"] = AdaBoostClassifier() # basic algorithm for boosting, a method by which can obtain one better from a large number of weak classifiers
 
classifiers["Bagging"] = BaggingClassifier() # Bagging (Bootstrap Aggregation) belongs to the procedures that aggregate a family of classifiers into one collective classifier, which is based on a majority vote.

from sklearn.metrics import classification_report, confusion_matrix

for clf_name in classifiers:
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Classification with: ",clf_name)
    print()

### learning with a specific algorithm
    
    clf = classifiers[clf_name]
    clf.fit(X_tr,y_tr) ### learning on training data
    
### prediction
    
    y_prediction = clf.predict(X_te) ### y_prediction as a result of classifier prediction on the test file

    print(classification_report(y_te, y_prediction)) # comparing y_testing with y_prediction we determine how good the classifier is
    
    print(confusion_matrix(y_te, y_prediction))