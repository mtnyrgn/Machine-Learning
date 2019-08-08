# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:13:37 2019

@author: metin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("maaslar.csv")
x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values

#Decision Tree
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

#Random Forest Regressor

rf = RandomForestRegressor(n_estimators = 10,random_state=0)# n_estimator:how many decision tree we will use.
rf.fit(X,Y)
print("Random Forest:",rf.predict(6.6))
plt.scatter(X,Y,color="red")
plt.plot(X,rf.predict(X),color="blue")
plt.plot(Z,rf.predict(X),color="green")
plt.plot(K,rf.predict(X),color="yellow")
plt.show()

#Lets calculat r2 score...
#for X
print("R2 value of Decision Tree:",r2_score(Y,r_dt.predict(X)))
print("R2 value of  Random Forest:",r2_score(Y,rf.predict(X)))

#for Z
print("R2 value of Decision Tree:",r2_score(Y,r_dt.predict(Z)))
print("R2 value of  Random Forest:",r2_score(Y,rf.predict(Z)))

#for K
print("R2 value of Decision Tree:",r2_score(Y,r_dt.predict(K)))
print("R2 value of  Random Forest:",r2_score(Y,rf.predict(K)))

"""
Decision tree looking perfectly algorithm according to r2 score but we know that decision tree produce same value for same range.
"""


