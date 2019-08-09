# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 02:10:11 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("veriler.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
y_predict = dt.predict(X_test)
cm = confusion_matrix(y_test,y_predict)
print(cm)

rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(X_train,y_train)
y_predict2 = rfc.predict(X_test)
cm_rfc = confusion_matrix(y_test,y_predict2)
print(cm_rfc)

