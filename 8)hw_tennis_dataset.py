# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:46:07 2019

@author: metin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

data = pd.read_csv("hw_tennis.csv")
print(data)

data2 = data.apply(LabelEncoder().fit_transform) #Numeric değerlerde encoding edildi.Bunu düzeltmeliyim

c = data2.iloc[:,:1]
ohe = OneHotEncoder()
c = ohe.fit_transform(c).toarray() #ilk sütun 3 farklı değer alıyordu.Bunları sütun haline çevirdim
print(c)

weather = pd.DataFrame(data = c,index=range(14),columns=["O","R","S"])
humidity = data.iloc[:,2:3]
windy_play = data2.iloc[:,3:]
outlook_temp = data2.iloc[:,0:2]

concat = pd.concat([weather,windy_play],axis=1)
concat = pd.concat([concat,outlook_temp],axis=1)

x_train,x_test,y_train,y_test = train_test_split(concat,humidity,test_size=0.33,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

#backward elimination
X = np.append(arr = np.ones((14,1)).astype(int),values=concat.iloc[:,:-1],axis=1)
X_list = concat.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = concat.iloc[:,-1:], exog =X_list)
r = r_ols.fit()
print(r.summary())


concat = concat.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=concat.iloc[:,:-1], axis=1 )
X_list = concat.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = concat.iloc[:,-1:], exog =X_list)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)












