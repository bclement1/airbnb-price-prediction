import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #stratify
import preprocessing 
#import os 
#os.system("preprocessing.py")

(X_train, y_train, X_test, y_test, df_identifiers) = preprocessing.run_preprocessing()
print("Nombre de données = ", X_train.shape[0])


nb_feature = 122

#parameters = {"solver" : ['adam', 'lbfgs', 'sgd'], "activation": ['identity', 'logistic', 'tanh', 'relu']}

#parameters = {"alpha": [1e-4, 1e-5, 1e-6], "learning_rate_init": [0.001, 0.0001, 0.00001], "max_iter": [1000, 10000, 100000]}

parameters = {"hidden_layer_sizes": [(256, 512, 256, 128, 16), (256, 512, 256, 128, 64, 16), (256, 128, 64, 16)]}

solver = 'adam'
alpha_regularization = 1e-5
max_iteration = 10000
learning_rate = 0.001

#hidden_layer_sizes = (nb_feature, 256, 128, 64, 16, 1) # TrainL = 0.49, TestL = 0.29
#hidden_layer_sizes = (nb_feature, 256, 64, 16, 1) # TrainL = 0.44, TestL = 0.31
#hidden_layer_sizes = (nb_feature, 256, 16, 1) # TrainL = 0.45, TestL = 0.30
#hidden_layer_sizes = (nb_feature, 256, 512, 256, 128, 64, 16, 1)
#hidden_layer_sizes = (nb_feature, 128, 1)

#hidden_layer_sizes = (256, 512, 256, 128, 16)
hidden_layer_sizes = (128, 256, 128)

X_train = X_train.to_numpy()
#y_train = y_train.to_numpy()

X_test = X_test.to_numpy()
#y_test = y_test.to_numpy()

regressor = MLPRegressor(solver=solver,
                alpha=alpha_regularization,     # used for regularization, ovoiding overfitting by penalizing large magnitudes
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=learning_rate,
                activation='relu', # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’ (default)
                max_iter=max_iteration,
                early_stopping=True,
                random_state=42)

print(regressor.get_params())

clf = GridSearchCV(regressor, parameters, scoring="neg_root_mean_squared_error", verbose=10, return_train_score=True)
clf.fit(X_train, y_train)
print("Résultats de l'optimisation = ", clf.cv_results_)
print("Score de test = ", clf.score(X_test, y_test))


plt.hist(clf.predict(X_test), bins=100, density=True, alpha=0.5)
plt.hist(y_test, bins=100, density=True, alpha=0.5)
plt.show()