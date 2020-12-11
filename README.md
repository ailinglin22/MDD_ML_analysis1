# MDD-project
This is the machine learnig implementation on MDD depression project. It contains mainly two tasks:
1. Supervised learning on the control and MDD groups (Classification).
* Random Forest (RF)
* Support Vector Machine (SVM)
* SHapley Additive exPlanations (SHAP) 
2. Unsupervised learning on the MDD subgroups (Clustering). 
* P-Value (Linear Regression)
* Hierarchical Clustering 
## Prerequisites
- [Python3](https://www.python.org/)
- [Numpy](https://numpy.org/)
- [scikit learn](https://scikit-learn.org)
- [Matplotlib](https://matplotlib.org/)

## Workflow
![](https://github.com/linbrainlab/MDD-project/blob/main/imgs/MDD_workflow.png)

## Preliminary Experiment

###Clustering
sticsa_somatic: (total=32)
group0 (n=19) avg. age:  35.89		avg. score: 18.16
group1 (n=13) avg. age:  38.46		avg. score: 13.85
age p-value= 0.5475660230583064
score p-value= 0.001965779014599557

![](https://github.com/linbrainlab/MDD-project/blob/main/imgs/Fig1.png)
![](https://github.com/linbrainlab/MDD-project/blob/main/imgs/Fig2.png)
![](https://github.com/linbrainlab/MDD-project/blob/main/imgs/Fig3.png)

