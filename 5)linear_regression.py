# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:40:49 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split

data = pd.read_csv('satislar.csv') 
print(data) 

months = data[['Aylar']]
print(months) 

sellings = data[['Satislar']]
print(sellings)

x_train,x_test,y_train,y_test = train_test_split(months,sellings,test_size=0.33,random_state=0)
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train)
predict = lr.predict(X_test)
print(predict)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Sellings to months")
plt.xlabel("Months")
plt.ylabel("Sellings")
