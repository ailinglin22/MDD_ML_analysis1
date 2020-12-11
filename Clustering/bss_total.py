import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering 
import numpy as np
from sklearn.decomposition import PCA 

data=pd.read_csv('/Users/xinxing/Desktop/MDD/New data/MDD_test/MDD_score/MDD_bss_total.csv').fillna(0)



#MDD only:
data=data[data.Status==0]
df=data[["Age","bss_total"]] 
df=df.reset_index(drop=True)

'''
data=data[['Status',"Age","RGprecentral","LGprecentral", "Rparsopercularis","N_Left-Lateral-Ventricle",
           "LGparstriangularis", "RGlingual", "N_Left-Cerebellum-Cortex", "Rinsula",
           "Linferiorparietal", "LGpostcentral","N_Right-Cerebellum-Cortex","Rpericalcarine",
           "RGsuperiorparietal","Rsuperiortemporal"]] 
index=["Age","RGprecentral","LGprecentral", "Rparsopercularis","N_Left-Lateral-Ventricle",
           "LGparstriangularis", "RGlingual", "N_Left-Cerebellum-Cortex", "Rinsula",
           "Linferiorparietal", "LGpostcentral","N_Right-Cerebellum-Cortex","Rpericalcarine",
           "RGsuperiorparietal","Rsuperiortemporal"]
'''

data=data[['Status',"Rprecentral","LGcaudalanteriorcingulate", "Rtransversetemporal","Rsupramarginal"]]

index=["Rprecentral","LGcaudalanteriorcingulate", "Rtransversetemporal","Rsupramarginal"]

y=data.Status
 
X=data.drop('Status',axis=1)
X=X.to_numpy()


#normalized standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X=pd.DataFrame(X,columns=index)
 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X)
# Calculate the distance between each sample
#Z = hierarchy.linkage(X_principal, 'ward')
Z = hierarchy.linkage(X, 'ward')
 
# Set the colour of the cluster here:
hierarchy.set_link_color_palette(['r', 'b'])
 
# Make the dendrogram and give the colour above threshold
hierarchy.dendrogram(Z, color_threshold=7, above_threshold_color='grey')
 
# Add horizontal line.
plt.axhline(y=7, c='black', lw=2, linestyle='dashed')

#from scipy.cluster.hierarchy import fcluster
#d=shc.linkage(X_principal, method ='ward')

ac2 = AgglomerativeClustering(n_clusters = 2,compute_full_tree=True)

# Visualizing the clustering 
plt.figure(figsize =(6, 6)) 

color=['b','r']

for i in range(X.shape[0]):
    plt.scatter(X_principal[i,0], X_principal[i,1],  
           c = color[ac2.fit_predict(X)[i]], cmap ='rainbow') 

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.axis('off')

'''    
for i in range (35):
    
    plt.annotate(z[i],(X_principal[i, 0], X_principal[i, 1]))
'''

labels=ac2.fit_predict(X)
DX=pd.DataFrame(labels,columns=["label"])
df_concat = pd.concat([df, DX], axis=1)

print(("total samples is: ") + str(df_concat.shape[0]))

print(("group 0 number is:  ") + str(df_concat.shape[0] - np.sum(labels)))

print(("group 1 number is:  ") + str(np.sum(labels)))

print(("Group 0 avg. age is: ") + str(df_concat[df_concat.label==0]["Age"].mean()))

print(("Group 1 avg. age is: ") + str(df_concat[df_concat.label==1]["Age"].mean()))

print(("Group 0 avg. score is: ") + str(df_concat[df_concat.label==0]["bss_total"].mean()))

print(("Group 1 avg. score is: ") + str(df_concat[df_concat.label==1]["bss_total"].mean()))
from scipy.stats import ttest_ind
stat, p = ttest_ind(df_concat[df_concat.label==0].Age,df_concat[df_concat.label==1].Age)
print("Age p-value is: "+ str(p))

stat_score, p_score = ttest_ind(df_concat[df_concat.label==0].bss_total,df_concat[df_concat.label==1].bss_total)
print("Score p-value is:" +str(p_score) )

import seaborn as sns

lut = dict(zip(set(labels), "br"))
row_colors=pd.DataFrame(labels)[0].map(lut)

sns.clustermap(X,method='ward',row_colors=row_colors)
#plt.axvline(x=2,c='black', lw=2, linestyle='dashed')
plt.show() 

