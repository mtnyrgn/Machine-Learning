# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:40:49 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split

data = pd.read_csv('eksikveriler.csv') 
print(data) #show data on console

height = data[['boy']]
print(height)

height_weight = data[['boy','kilo']]
print(height_weight)

#missing values


imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)

age = data.iloc[:,1:4].values
print(age)

imputer =imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])
print(age)

country = data.iloc[:,0:1].values
print(country)
 
#Encoder: Categorical->Numerical
le = LabelEncoder()
country[:,0] = le.fit_transform(country[:,0])
print(country)

ohe = OneHotEncoder(categorical_features="all")
country = ohe.fit_transform(country).toarray()
print(country)
# Numpy Array -> Data Frame
country_result = pd.DataFrame(data = country,index=range(22),columns=["fr","tr","usa"])
print(country_result)
age_result = pd.DataFrame(data = age,index=range(22),columns=["height","weight","age"])
print(country_result)
sex = data.iloc[:,-1].values
print(sex)
sex_result = pd.DataFrame(data = sex,index = range(22),columns=["sex"])

#compose of data frames
concat1 = pd.concat([country_result,age_result],axis=1)
concat2 = pd.concat([concat1,sex_result],axis=1)
#tahmin etmek istediğim şey kadın/erkek olması.Bu yüzden kadın erkek sütununu diğerleriinden ayırdım.benim için Y olacak.

x_train,x_test,y_train,y_test=train_test_split(concat1,sex_result,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
#Normalizasyon tercih edilmeme nedeni outlier değerler.10000 gibi büyük değer olsaydı diğer değerler 0 temsil edilecekti.
#bu yüzden standadization daha çok tercih ediliyor.
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

