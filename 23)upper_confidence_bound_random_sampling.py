# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 02:04:17 2019

@author: metin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv('Ads_CTR_Optimisation.csv')
    
"""Random Sampling
selected = []
N = 10000 #number of tuples
d = 10 #number of advertisement
total = 0
for n in range(0,N):
    adv = random.randrange(d)
    selected.append(adv)
    pleasure = data.values[n,adv]
    total +=pleasure
    
plt.hist(selected)
plt.show()
"""
selected = []
N = 10000 #transaction
d = 10 # number of adv
total_pleasure = 0 #total pleasure"
clicked =[0]*d #clicked at the moment
pleasures =[0] * d #clicked at the moment
total_pleasure = 0
for n in range(1,N):
    adv = 0
    max_ucb = 0
    for i in range(0,d):
        if(clicked[i] > 0):
            average_pleasure = pleasures[i]/clicked[i]
            delta = math.sqrt(3/2*math.log(n)/clicked[i])
            ucb = average_pleasure + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            adv = i
    selected.append(adv)
    clicked[adv] = clicked[adv] + 1
    pleasure = data.values[n,adv]
    pleasures[adv] = pleasures[adv] + pleasure
    total_pleasure = total_pleasure + pleasure
print("Total Pleasure")
print(total_pleasure)
plt.hist(selected)
plt.show()
    