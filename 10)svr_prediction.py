# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:45:30 2019

@author: metin
"""

#We must use scaler.

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

#SVR
scaler1 = StandardScaler()
x_scaling = scaler1.fit_transform(X)
scaler2 = StandardScaler()
y_scaling = scaler2.fit_transform(Y)

from sklearn.svm import SVR
#Specifies the kernel type to be used in the algorithm. 
#It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used.
svr_reg = SVR(kernel='rbf') #kernel->polynomial,linear,rbf,...etc
svr_reg.fit(x_scaling,y_scaling)
plt.scatter(x_scaling,y_scaling,color="yellow")
plt.plot(x_scaling,svr_reg.predict(x_scaling),color="green")
plt.show()
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))

#kernel ->sigmoid
svr_reg2 = SVR(kernel='sigmoid') #kernel->polynomial,linear,rbf,...etc
svr_reg2.fit(x_scaling,y_scaling)
plt.scatter(x_scaling,y_scaling,color="blue")
plt.plot(x_scaling,svr_reg2.predict(x_scaling),color="red")
plt.show()
print(svr_reg2.predict(11))
print(svr_reg2.predict(6.6))

#kernel ->poly
svr_reg3 = SVR(kernel='poly') #kernel->polynomial,linear,rbf,...etc
svr_reg3.fit(x_scaling,y_scaling)
plt.scatter(x_scaling,y_scaling,color="green")
plt.plot(x_scaling,svr_reg3.predict(x_scaling),color="blue")
plt.show()
print(svr_reg3.predict(11))
print(svr_reg3.predict(6.6))