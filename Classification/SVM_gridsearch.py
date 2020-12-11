#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 01:01:46 2020

@author: xinxing
"""

#Features without image features:


import matplotlib.pyplot as plt

from sklearn import svm
from scipy import interp
import numpy as np
import pandas as pd
from itertools import cycle

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
#from imblearn.combine import SMOTEENN
#from imblearn.over_sampling import SMOTE 



#original data
#data=pd.read_csv('/Users/xinxing/Documents/XIN/Work/DrLin/ADNI/NewData/combined_norm_features_final.csv',encoding='mac_greek').fillna(0)

#corviate balancing data
data=pd.read_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_data_clf.csv').fillna(0)


#data=data.drop("Status", axis=1)
#data=data[data.Status==0]



df=data

#df_MCI=pd.concat([df_LMCI, df_EMCI])
#df['DX'].value_counts()

#Try the original:

y=df.Status
X=df.drop(['Age','7T ID', 'XNAT','Status', 'Gender'],axis=1)


'''
#top 20
X=X[["LGsuperiorparietal","LGentorhinal","Rpostcentral", "RGpostcentral","LGcaudalanteriorcingulate",
     "RGinferiorparietal", "N_Right-Thalamus-Proper", "Linferiorparietal", "N_Right-Inf-Lat-Vent","RGpericalcarine", 
      "Lprecentral", "Rsuperiorfrontal", "Linsula", "Rcuneus","RGsuperiorparietal",
      "LGfusiform", "LGprecentral", "LGpostcentral", "Rparacentral","LGsuperiortemporal"
      ]]

#top 15
X=X[["LGsuperiorparietal","LGentorhinal","Rpostcentral", "RGpostcentral","LGcaudalanteriorcingulate",
     "RGinferiorparietal", "N_Right-Thalamus-Proper", "Linferiorparietal", "N_Right-Inf-Lat-Vent","RGpericalcarine", 
      "Lprecentral", "Rsuperiorfrontal", "Linsula", "Rcuneus","LGfusiform", 
      ]]
'''
#top 10
X=X[["LGsuperiorparietal","LGentorhinal","Rpostcentral", "RGpostcentral","LGcaudalanteriorcingulate",
     "RGinferiorparietal", "N_Right-Thalamus-Proper", "Linferiorparietal", "Rsuperiorfrontal","RGpericalcarine"
        
      ]]

#b=np.mean(importance, axis=0)
#feature_importances = pd.DataFrame(b,index = X_1.columns, columns=['importance']).sort_values('importance',ascending=False)



numFeature=X.shape[1]
#X=X.as_matrix()
y=y.ravel()
X_1=X

#normalized standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)


#clf=RandomForestClassifier(n_estimators=100,criterion='entropy')#,class_weight='balanced')
#clf=svm.SVC(kernel='rbf',C=1000, gamma=0.01,coef0=0.01,probability=True)
#cv = StratifiedKFold(n_splits=3)
result= []
res=[]
f1=[]
from sklearn.svm import SVC
#SVM
from sklearn.model_selection import GridSearchCV
#kernel=['rbf']
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)


clf = SVC()
grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)
print("Accuracy:"+ str(grid.best_score_))




