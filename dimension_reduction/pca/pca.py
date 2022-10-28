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
import seaborn as sns

## Preprocessing
data = preprocess.load_data()
(X_train, y_train, X_test, y_test, df_identifiers) = preprocess.preprocessing(data)

# Find the principal components axes
pca = PCA()
X_reduced = pca.fit_transform(X_train)
#print(pd.DataFrame(pca.components_).loc[:10,:5])

# Get the optimal number of PCA
def show_mse_by_number_of_pca(fold_nb = 10, nb_of_pca_to_show = 100):
    n = len(X_reduced)
    kf = model_selection.KFold(n_splits=10)
    regr = LinearRegression()
    mse = []
    
    # Compute MSE with only the intercept (no principal components in regression)
    score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf, scoring='neg_mean_squared_error').mean()    
    mse.append(score)
    # Compute MSE using CV for the 50 principal components
    for i in range(1,nb_of_pca_to_show):
        score = -1*model_selection.cross_val_score(regr, X_reduced[:,:i], y_train.ravel(), cv=kf, scoring='neg_mean_squared_error').mean()
        mse.append(score)
    plt.plot(mse, '-v')
    plt.xlabel('Number of principal components in regression')
    plt.ylabel('MSE')
    plt.xlim(left=-1);  
    plt.show()  
    # We can see 2 discontinuity in the value of the mse, one at 5 principle component, one at 27 principal component
    # We keep only 27 principal components out of 122

def load_optimal_pca(X_preprocess, component_nb = 27):
    pca_opti = PCA(n_components = component_nb)
    pca_opti.fit_transform(X_train)    
    return pca_opti

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

# Effect of variables on each components
def show_effect_of_features_on_pc(pcamodel, X_train, nb_feature_to_show = 50):
    ax = sns.heatmap(pcamodel.components_[:,:nb_feature_to_show],
                    cmap='YlGnBu',
                    yticklabels=[ "PCA"+str(X_train) for X_train in range(1,pcamodel.n_components_+1)],
                    xticklabels=list(X_train.iloc[: , :nb_feature_to_show].columns),
                    cbar_kws={"orientation": "vertical"})
    plt.yticks(rotation=0) 
    ax.set_aspect("equal")
    plt.title('The effect of features on each components')
    plt.show()

if __name__ == "__main__":
    show_mse_by_number_of_pca()
    pcamodel = load_optimal_pca(X_reduced)
    get_explained_variance(pcamodel)
    show_effect_of_features_on_pc(pcamodel, X_train)
