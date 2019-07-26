# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:40:49 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('eksikveriler.csv') 
print(data) #show data on console

height = data[['boy']]
print(height)

height_weight = data[['boy','kilo']]
print(height_weight)

#missing values
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)

age = data.iloc[:,1:4].values
print(age)

imputer =imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])
print(age)

