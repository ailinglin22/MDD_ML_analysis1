#Features without image features:


import matplotlib.pyplot as plt

from sklearn import svm
from scipy import interp
import numpy as np
import pandas as pd
from itertools import cycle

from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor)
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
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 250)
result= []
importance=np.zeros((500,numFeature))
res=[]
f1=[]

i = 0
j=0 
for i in range(100):
    clf = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=8, max_features='auto',max_depth=50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    for train, test in cv.split(X, y):
    
        #sm= SMOTEENN(random_state=44)
        #X_res,y_res=sm.fit_sample(X[train],y[train])
        #probas_ = clf.fit(X_res, y_res).predict_proba(X[test])#The smote upsampling
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test]) #original
        accuracy=clf.predict(X[test])
        res.append(clf.score(X[test],y[test]))
        #print(('accuracy: ')+str(clf.score(X[test],y[test])))
        y_pred=clf.predict(X[test])
        f1.append(f1_score(y[test], y_pred, average='weighted'))
        #print(('F1 socre: ')+str(f1_score(y[test], y_pred, average='weighted')))
        importance[j,:]=clf.feature_importances_
        
        j += 1
    i += 1
b=np.mean(importance, axis=0)
feature_importances = pd.DataFrame(b,index = X_1.columns, columns=['importance']).sort_values('importance',ascending=False)    
print("The acc is: "+str(np.mean(res)))
print("The f1 is: " +str(np.mean(f1)))
print("The feature ranking is:")
print(feature_importances[:30])

'''
from sklearn.model_selection import train_test_split
#clf=RandomForestClassifier(n_estimators=100,criterion='entropy')#,class_weight='balanced')
#clf=svm.SVC(kernel='rbf',C=1000, gamma=0.01,coef0=0.01,probability=True)
#cv = StratifiedKFold(n_splits=3)
tprs = []
aucs = []

result= []
importance=np.zeros((1,numFeature))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)

reg=RandomForestRegressor(n_estimators=10)

#reg.fit(X_train,y_train) 
#y = reg.predict(X_test)
reg.fit(X, y)

#error=np.sum(y-y_test)

importance =reg.feature_importances_


feature_importances = pd.DataFrame(importance,index = X_1.columns, columns=['importance']).sort_values('importance',ascending=False)    

print("The feature ranking is:")
print(feature_importances)
'''
