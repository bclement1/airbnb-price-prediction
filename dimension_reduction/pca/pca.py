# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 2022

@author: Mathian
"""
#Import
import pandas as pd
import sys
sys.path.append("preprocessing/")
from preprocessing import preprocessing
import preprocessing as preprocess
import numpy as np
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')  # Necessary to use Matplotlib on my computer
import matplotlib.pyplot as plt

## Preprocessing
data = preprocess.load_data()
(X_train, y_train, X_test, y_test, df_identifiers) = preprocess.preprocessing(data)

# Find the PCA
pca = PCA()
X_reduced = pca.fit_transform(X_train)
#print(pd.DataFrame(pca.components_.T).loc[:5,:5])

# Get the optimal number of PCA
def get_mse(fold_nb = 10, nb_of_pca_to_show = 15):
    n = len(X_reduced)
    kf = model_selection.KFold(n_splits=10)
    regr = LinearRegression()
    mse = []
    
    # Compute MSE with only the intercept (no principal components in regression)
    score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf, scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    # Compute MSE using CV for the 15 principal components
    for i in range(1,nb_of_pca_to_show):
        score = -1*model_selection.cross_val_score(regr, X_reduced[:,:i], y_train.ravel(), cv=kf, scoring='neg_mean_squared_error').mean()
        mse.append(score)
    plt.plot(mse, '-v')
    plt.xlabel('Number of principal components in regression')
    plt.ylabel('MSE')
    plt.xlim(left=-1);  
    plt.show()  
    # We can see 2 discontinuity in the value of the mse, one at 5 principle component, one at 25 principal component
    # We keep only 5 principal component

def load_optimal_pca(X_preprocess, component_nb = 5):
    pca = PCA(n_components = component_nb)
    pca.fit_transform(X_preprocess)    
    return pca

def get_explained_variance(pcamodel):
    plt.bar(range(1,len(pcamodel.explained_variance_ratio_ )+1),pcamodel.explained_variance_ratio_ )
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Number of components')
    plt.plot(range(1,len(pcamodel.explained_variance_ratio_ )+1),
            np.cumsum(pcamodel.explained_variance_ratio_ ),
            c='red',
            label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    get_mse()
    pcamodel = load_optimal_pca(X_reduced)
    get_explained_variance(pcamodel)
