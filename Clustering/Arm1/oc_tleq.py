#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:05:26 2021

@author: xinxing
"""


import numpy as np

import pandas as pd
import statsmodels.api as sm


data=pd.read_excel('/Users/xinxing/Documents/MDD/New data/MDD_test/May_03_2021/MDD_only_data_Gender.xlsx')
#data=pd.read_csv('/Users/xinxing/Documents/MDD/New data/MDD_test/May_03_2021/MDD_only_data_Gender.csv')
data=data[data.Status==0]

item='oc_tleq'
data=data.dropna(subset=[item])

#data=pd.read_csv("/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_ce_tleq.csv")

#data=data[data.Status==0]
drop_list=['7T ID','XNAT','Status','Label','Height','Weight','BMI','Race','Ethnicity',
           'Education','Employment Status','Household Income']
scorelist=['madrs_sum','pss_score','btq_score','lsc_score','ce_tleq','oc_tleq',
           'qids_score','rrs_total','shaps_score_1','shaps_score_2','bss_total',
           'sticsa_somatic','sticsa_cognitive']
df=data.drop(drop_list,axis=1)


y=df[item]

X=df.drop(scorelist, axis=1)

index=X.columns
index=index.to_list()
index.pop(0)
index.pop(0)


X_1=X

numFeature=X.shape[1]

X_d=X.to_numpy()

list_pvalue=[]

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

res_pvalue=pvalue[pvalue.p_value<0.05]
#print(res.T[:20])

#res.T[:30].to_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_bss_total_ranking.csv') 

#res_pvalue.to_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_ce_tleq_ranking_p.csv') 

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
savepath='/Users/xinxing/Documents/MDD/New data/MDD_test/May_03_2021/MDD_'+item+'_ranking_p.csv'
res_pvalue_1.to_csv(savepath) 
index=res_pvalue_1.index.to_list()

import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 

dfnew=data[[item,"Age","Gender"]] 
dfnew=dfnew.reset_index(drop=True)

newlist=index+['Status']

datanew=data[newlist]

y_new=datanew.Status
 
X_new=datanew.drop('Status',axis=1)

#normalized standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_new)
X_new=scaler.transform(X_new)
X_new=pd.DataFrame(X_new,columns=index)


pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_new)
# Calculate the distance between each sample
#Z = hierarchy.linkage(X_principal, 'ward')
Z = hierarchy.linkage(X_new, 'ward')
 
# Set the colour of the cluster here:
hierarchy.set_link_color_palette(['r', 'b'])
 
# Make the dendrogram and give the colour above threshold
hierarchy.dendrogram(Z, color_threshold=14, above_threshold_color='grey')
 
# Add horizontal line.
plt.axhline(y=14, c='black', lw=2, linestyle='dashed')

#from scipy.cluster.hierarchy import fcluster
#d=shc.linkage(X_principal, method ='ward')

ac2 = AgglomerativeClustering(n_clusters = 2,compute_full_tree=True)

# Visualizing the clustering 
plt.figure(figsize =(6, 6)) 

color=['b','r']

for i in range(X_new.shape[0]):
    plt.scatter(X_principal[i,0], X_principal[i,1],  
           c = color[ac2.fit_predict(X_new)[i]], cmap ='rainbow') 

'''    
for i in range (35):
    
    plt.annotate(z[i],(X_principal[i, 0], X_principal[i, 1]))
'''
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.axis('off')
labels=ac2.fit_predict(X_new)

DX=pd.DataFrame(labels,columns=["label"])
df_concat = pd.concat([dfnew, DX], axis=1)

print(("total samples is: ") + str(df_concat.shape[0]))

print(("group 0 number is:  ") + str(df_concat.shape[0] - np.sum(labels)))

print(("group 1 number is:  ") + str(np.sum(labels)))

print(("Group 0 avg. age is: ") + str(df_concat[df_concat.label==0]["Age"].mean()))

print(("Group 1 avg. age is: ") + str(df_concat[df_concat.label==1]["Age"].mean()))

print(("Group 0 avg. score is: ") + str(df_concat[df_concat.label==0][item].mean()))

print(("Group 1 avg. score is: ") + str(df_concat[df_concat.label==1][item].mean()))

from scipy.stats import ttest_ind
stat, p = ttest_ind(df_concat[df_concat.label==0].Age,df_concat[df_concat.label==1].Age)

print("Age p-value is: "+str(p))

stat_score, p_score = ttest_ind(df_concat[df_concat.label==0][item],df_concat[df_concat.label==1][item])
print("Score p-value is:" +str(p_score) )

stat, p = ttest_ind(df_concat[df_concat.label==0].Gender,df_concat[df_concat.label==1].Gender)
print("Gender p-value is: "+ str(p))

import seaborn as sns

lut = dict(zip(set(labels), "br"))

row_colors=pd.DataFrame(labels)[0].map(lut)

sns.clustermap(X_new,method='ward',row_colors=row_colors)
#plt.axvline(x=2,c='black', lw=2, linestyle='dashed')
plt.show() 



'''
import pickle
with open("/Users/xinxing/Documents/MDD/New data/MDD_test/May_03_2021/madrs_sum_index.txt", "wb") as fp:
    pickle.dump(index, fp)
'''