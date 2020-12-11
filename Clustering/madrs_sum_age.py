#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:25:40 2020

@author: xinxing
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:23:56 2020

@author: xinxing
"""

import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.linear_model import LinearRegression


data=pd.read_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_madrs_sum.csv').fillna(0)
data=data[data.Status==0]

df=data[['Status',"Age","LGfusiform", "RGentorhinal", "LGinferiorparietal","RGfusiform",
           "RGparstriangularis", "Rparsopercularis", "LGrostralmiddlefrontal", "Lmedialorbitofrontal",
           "LGparsopercularis", "RGmedialorbitofrontal", "LGsuperiorfrontal", "LGsuperiorparietal", "Lfrontalpole"]] 

y=df.Age
X=df.drop(["Age","Status"], axis=1)

X_1=X
reg = LinearRegression().fit(X, y)




