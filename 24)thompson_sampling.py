# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:23:25 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000 #transaction
d = 10 # number of adv
selected = []
total_pleasure = 0 #total pleasure"
ones = [0] * d
zeros = [0] * d
for n in range(1,N):
    ad = 0
    max_th = 0  
    for i in range(0,d):
        rasbeta = random.betavariate(ones[i]+1,zeros[i]+1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    selected.append(ad)
    pleasure = data.values[n,ad]
    if pleasure == 1:
        ones[ad] = ones[ad] + 1
    else:
        zeros[ad] = zeros[ad] + 1
    total_pleasure = total_pleasure + pleasure
print("Total Pleasure")
print(total_pleasure)
plt.hist(selected)
plt.show()
    