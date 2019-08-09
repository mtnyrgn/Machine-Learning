# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:46:54 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("veriler.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_predict = logr.predict(X_test)
print(y_predict)
print(y_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_predict)
print(cm)




