#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import datetime


# In[2]:


import preprocessing.preprocessing as preprocess


# In[3]:


pd.options.display.max_columns = None


# In[4]:


data = preprocess.load_data()
(X_train, y_train, X_test, y_test, df_identifiers) = preprocess.preprocessing(data)


# # xgboost

# In[5]:


datetime_columns = ["Host Since", "First Review", "Last Review"]
for col in datetime_columns:
    X_train[col + "_year"] = X_train[col].apply(lambda x: x.year)
    X_train[col + "_month"] = X_train[col].apply(lambda x: x.month)
    X_train[col + "_day"] = X_train[col].apply(lambda x: x.day)
X_train.drop(columns=datetime_columns, inplace=True)


# In[6]:


datetime_columns = ["Host Since", "First Review", "Last Review"]
for col in datetime_columns:
    X_test[col + "_year"] = X_test[col].apply(lambda x: x.year)
    X_test[col + "_month"] = X_test[col].apply(lambda x: x.month)
    X_test[col + "_day"] = X_test[col].apply(lambda x: x.day)
X_test.drop(columns=datetime_columns, inplace=True)


# In[7]:


na_index = y_train[y_train.isna()].index


# In[8]:


X_train.drop(na_index, inplace=True, errors="ignore")


# In[9]:


y_train.drop(na_index, inplace=True, errors="ignore")


# In[10]:


import xgboost as xg

model = xg.XGBRegressor(
    learning_rate=0.001,
    n_estimators=6000,
    max_depth=4,
    min_child_weight=0,
    gamma=0.6,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.00006,
    random_state=42,
)

model.fit(X_train, y_train)

yhat = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))


# In[11]:


save_file_path = "logs/xgboost/{}/".format(
    datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(save_file_path, exist_ok=True)
model.save_model(save_file_path + "/model.json")

# model_xgb_2 = xgb.Booster()
# model_xgb_2.load_model("model.json")


# In[13]:


estimator = xg.XGBRegressor(objective="reg:squarederror", nthread=4, seed=42)

parameters = {
    "max_depth": [3, 4],
    "learning_rate": [0.1, 0.05],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_


grid_search.best_params_


# # Decision Tree

# In[13]:


from sklearn import tree

model = tree.DecisionTreeRegressor()

model.fit(X_train, y_train)

yhat = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))


# In[ ]:


estimator = tree.DecisionTreeRegressor()

parameters = {
    "max_depth": range(2, 10, 1),
    "n_estimators": range(60, 220, 40),
    "learning_rate": [0.1, 0.01, 0.05],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_


grid_search.best_params


# # Bagging

# In[14]:


from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor


# In[15]:


model = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)

model.fit(X_train, y_train)

yhat = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))


# In[ ]:


estimator = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)

parameters = {
    "max_depth": range(2, 10, 1),
    "n_estimators": range(60, 220, 40),
    "learning_rate": [0.1, 0.01, 0.05],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_


grid_search.best_params


# # Boosting

# In[17]:


from sklearn.ensemble import GradientBoostingRegressor


# In[18]:


model = GradientBoostingRegressor()

model.fit(X_train, y_train)

yhat = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))


# In[ ]:


estimator = GradientBoostingRegressor()
parameters = {
    "max_depth": range(2, 10, 1),
    "n_estimators": range(60, 220, 40),
    "learning_rate": [0.1, 0.01, 0.05],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_


grid_search.best_params


# # Random Forest

# In[20]:


from sklearn.ensemble import RandomForestRegressor


# In[21]:


model = RandomForestRegressor()

model.fit(X_train, y_train)

yhat = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))


# In[ ]:


estimator = RandomForestRegressor()
parameters = {
    "max_depth": range(2, 10, 1),
    "n_estimators": range(60, 220, 40),
    "learning_rate": [0.1, 0.01, 0.05],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_

grid_search.best_params


# # Adaboost

# In[22]:


from sklearn.ensemble import AdaBoostRegressor


# In[23]:


model = AdaBoostRegressor()

model.fit(X_train, y_train)

yhat = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))


# In[24]:


estimator = AdaBoostRegressor()
parameters = {
    "max_depth": range(2, 10, 1),
    "n_estimators": range(60, 220, 40),
    "learning_rate": [0.1, 0.01, 0.05],
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring="roc_auc",
    n_jobs=10,
    cv=10,
    verbose=True,
)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_

grid_search.best_params


# In[ ]:
