# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:41:35 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm 

data = pd.read_csv("maaslar_yeni.csv")
x = data.iloc[:,2:5] #first and second columns can't provide anything about prediction.
y = data.iloc[:,5:6]

X = x.values
Y = y.values


#Linear Regression
lr = LinearRegression()
lr.fit(X,Y)
model = sm.OLS(lr.predict(X),Y)
print(model.fit().summary()) #Max p value in 'Puan' column
print("R2 score of Linear Regression:",r2_score(Y,lr.predict(X)))

#PolynomialRegression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(x_poly,Y)
model2 = sm.OLS(lr2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary()) #Max p value in 'Puan' column
print("R2 score of Polynomial Regression:",r2_score(Y,lr2.predict(poly_reg.fit_transform(X))))

#SVR
scaler1 = StandardScaler()
x_scaling = scaler1.fit_transform(X)
scaler2 = StandardScaler()
y_scaling = scaler2.fit_transform(Y)
svr = SVR(kernel='rbf')
svr.fit(x_scaling,y_scaling)
model3 =sm.OLS(svr.predict(x_scaling),x_scaling)
print(model3.fit().summary)#max p value in 'Puan' column
print("R2 score of SVR",r2_score(y_scaling,svr.predict(x_scaling)))

#Decision Tree
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X,Y)
model4 = sm.OLS(dt.predict(X),X)
print(model4.fit().summary())
print("R2 score of Decision Tree:",r2_score(Y,dt.predict(X)))

#Random Forest
rf = RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,Y)
model5 = sm.OLS(rf.predict(X),X)
print(model5.fit().summary())
print("R2 score of Random Forest",r2_score(Y,rf.predict(X)))













