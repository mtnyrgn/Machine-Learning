# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:50:51 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
data = pd.read_excel("Iris.xls")

x = data.iloc[:,:4].values
y = data.iloc[:,4:5].values
le = LabelEncoder()
y = le.fit_transform(y)
sc = StandardScaler()
x = sc.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#Decisiion Tree
dt = DecisionTreeClassifier(criterion="entropy",random_state=0)
dt.fit(x_train,y_train)
y_predict = dt.predict(x_test)
cm = confusion_matrix(y_test,y_predict)
print("---------Confusion Matrix DT-------------")
print(cm)
print("--------------------------------------")

r_ols = sm.OLS(endog=y,exog = x)
r = r_ols.fit().summary()
print(r)

#Random Forest 
rf = RandomForestClassifier(n_estimators=15,criterion="gini",random_state=0)
rf.fit(x_train,y_train)
y_prob = rf.predict_proba(x_test)
y_predict2 = rf.predict(x_test)
cm = confusion_matrix(y_test,y_predict2)
print("---------Confusion Matrix Random Forest-------------")
print(cm)
print("----------------------------------------------------")

#KNN
knn = KNeighborsClassifier(n_neighbors=10,metric='minkowski')
knn.fit(x_train,y_train)
y_predict3 = knn.predict(x_test)
cm2 = confusion_matrix(y_test,y_predict3)
print("---------Confusion Matrix KNN-------------")
print(cm2)
print("--------------------------------------")

#LogistricRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_predict4 = lr.predict(x_test)
cm3 = confusion_matrix(y_test,y_predict4)
print("---------Confusion Matrix LR-------------")
print(cm3)
print("-----------------------------------------")

#SVM 
svm = SVC(kernel='rbf')
svm.fit(x_train,y_train)
y_predict5 = svm.predict(x_test)
cm4 = confusion_matrix(y_test,y_predict5)
print("---------Confusion Matrix SVM-------------")
print(cm4)
print("-----------------------------------------")

fpr,tpr,thold = metrics.roc_curve(y_test,y_prob[:,0],pos_label=2)
print("------FPR,TPR,THOLD-----")
print(fpr)
print(tpr)
print(thold)
print("------------------------")





