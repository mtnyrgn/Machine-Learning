# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 23:22:20 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.cross_validation import train_test_split

data = pd.read_csv("maaslar.csv")

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.plot(X,r_dt.predict(Z),color="yellow")
plt.plot(X,r_dt.predict(Z),color="green")
plt.show()

print("Decision Tree:", r_dt.predict(11))
print("Decision Tree:",r_dt.predict(6.6))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10,random_state=0)# n_estimator:how many decision tree we will use.
rf.fit(X,Y)
print("Random Forest:",rf.predict(6.6))

plt.scatter(X,Y,color="red")
plt.plot(X,rf.predict(X),color="blue")
plt.plot(Z,rf.predict(X),color="green")
plt.plot(K,rf.predict(X),color="yellow")
plt.show()


