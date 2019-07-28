# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:50:07 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

data = pd.read_csv("maaslar.csv")

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values

#linear regression
lr = LinearRegression()
lr.fit(X,Y)
plt.scatter(X,Y,color="red")
plt.plot(x,lr.predict(x),color="blue")
plt.show()

#polynomial regression
#Degree=2
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_poly1 = poly.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(x_poly1,Y)
plt.scatter(X,Y,color="yellow")
plt.plot(x,lr2.predict(poly.fit_transform(X)),color="green")
plt.show()
#Degree = 4
poly2 = PolynomialFeatures(degree = 4)
x_poly2 = poly2.fit_transform(X)
lr3 = LinearRegression()
lr3.fit(x_poly2,Y)
plt.scatter(X,Y,color="red")
plt.plot(x,lr3.predict(poly2.fit_transform(X)))
plt.show()

#prediction
print(lr.predict(11)) #prediction on linear regression
print(lr2.predict(poly.fit_transform(11))) #prediction on poly regression degree=2
print(lr3.predict(poly2.fit_transform(11))) #prediction on poly regression degree=4