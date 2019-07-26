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

data = pd.read_csv('veriler.csv') 

age = data.iloc[:,1:4].values
#encoding for country
country = data.iloc[:,0:1].values
print(country)
le = LabelEncoder()
country[:,0]=le.fit_transform(country[:,0])
print(country)
ohe = OneHotEncoder(categorical_features="all")
country=ohe.fit_transform(country).toarray()
print(country)

#encoding for sex
sex = data.iloc[:,-1:].values
print(sex)
le = LabelEncoder()
sex[:,0]=le.fit_transform(country[:,0])
print(sex)
ohe = OneHotEncoder(categorical_features="all")
sex=ohe.fit_transform(sex).toarray()
print(sex)

# Numpy Array -> Data Frame
country_result = pd.DataFrame(data = country,index=range(22),columns=["fr","tr","usa"])
print(country_result)
age_result = pd.DataFrame(data = age,index=range(22),columns=["boy","kilo","yas"])
print(country_result)
sex_result = pd.DataFrame(data = sex[:,:1],index = range(22),columns=["cinsiyet"])
print(sex_result)
#compose of data frames
concat1 = pd.concat([country_result,age_result],axis=1)
concat2 = pd.concat([concat1,sex_result],axis=1)

x_train,x_test,y_train,y_test=train_test_split(concat1,sex_result,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

x_train.sort_index()
plt.plot(x_train.sort_index())

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)

#----------------------------------------------

height = concat2.iloc[:,3:4].values
left =concat2.iloc[:,:3].values
right = concat2.iloc[:,4:].values
left = pd.DataFrame(data = left,index=range(22),columns=["fr","tr","us"])
right = pd.DataFrame(data = right,index = range(22),columns=["weight","age","sex"])
new_data = pd.concat([right,left],axis=1)

x_train,x_test,y_train,y_test = train_test_split(new_data,height,test_size=0.33,random_state=0)

regressor2 =LinearRegression()
regressor2.fit(x_train,y_train)
y2_predict=regressor2.predict(x_test)

import statsmodels.formula.api as sm
#if axis = 0 ->row
X = np.append(arr = np.ones((22,1)).astype(int),values=new_data,axis=1)

X_list = new_data.iloc[:,[0,2,3,4,5]].values
r_ols = sm.OLS(endog = height,exog =X_list)
r = r_ols.fit()
print(r.summary())



