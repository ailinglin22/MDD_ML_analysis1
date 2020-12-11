#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:04:03 2020

@author: xinxing
"""

from scipy import stats

import numpy as np

import pandas as pd

feature=[
   'rrs_total', 'shaps_score_1', 'shaps_score_2', 'bss_total', 
                          'sticsa_somatic', 'sticsa_cognitive']

data=pd.read_csv("/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_qids_score.csv")

data=data[data.Status==0]
df=data.drop(['Status'],axis=1)


y=df.qids_score

X=df.drop('qids_score', axis=1)
index=X.columns
index=index.to_list()
index.pop(0)

X_1=X

numFeature=X.shape[1]

X_d=X.to_numpy()

#list_rvalue=[]
#list_rvalue_abs=[]

list_pvalue=[]

import statsmodels.api as sm

for i in range (len(index)):
    Y=sm.add_constant(X[[index[i]]])
    est = sm.OLS(y, Y)
    est2 = est.fit()
    #print(est2.summary())
    p_values = est2.summary2().tables[1]['P>|t|']
    list_pvalue.append(p_values[1])
#feature_importances = pd.DataFrame(list_rvalue_abs,index = X_1.columns, columns=['r_value']).sort_values('r_value',ascending=False) 

#feature = pd.DataFrame(list_rvalue,index = X_1.columns, columns=['r_value']) 
pvalue=pd.DataFrame(list_pvalue,index = index, columns=['p_value']) .sort_values('p_value',ascending=True)

#listnew=feature_importances.T.columns.tolist()

#res=feature.T[listnew]

#res_pvalue=pvalue.T[index]
res_pvalue=pvalue[pvalue.p_value<0.05]
#print(res.T[:20])

#res.T[:30].to_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_bss_total_ranking.csv') 

#res_pvalue.to_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_qids_score_ranking_p.csv') 

#print(res_pvalue)
list_pvalue2=[]
index2=res_pvalue.T.columns
index2=index2.to_list()
for i in range (len(index2)):
    Y1=sm.add_constant(X[[index2[i],"Age"]])
    est = sm.OLS(y, Y1)
    est2 = est.fit()
    #print(est2.summary())
    p_values = est2.summary2().tables[1]['P>|t|']
    list_pvalue2.append(p_values[1])
pvalue_1=pd.DataFrame(list_pvalue2,index = index2, columns=['p_value']) .sort_values('p_value',ascending=True)

res_pvalue_1=pvalue_1[pvalue_1.p_value<0.05]
res_pvalue_1.to_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_qids_score_ranking_p.csv') 