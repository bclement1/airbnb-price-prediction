#############################################################################
#############################################################################
#############################################################################
"""
                                ML Projet
3A Centrale Supelec
Date : 18/10/2022
Author : Tristan BASLER
Description :
    Using XGBOOST model to predict price for an AirBnB appartement
                                                                          """
#############################################################################
#############################################################################
#############################################################################

# Importation
import sys
sys.path.insert(0, "../") # To insert the package on the path
import numpy as np
import os
import datetime
import preprocessing.preprocessing as preprocess
from sklearn.metrics import mean_squared_error
import xgboost as xg
from Tree_Method.utils import writing_description

# Params definition
FILL_NA_METHOD = None
ENCODER_METHOD = "LabelEncoder"
DATEITME_TREATMENT = 'Linearization'

# Save Folder initialization
save_file_path = "../logs/xgboost/{}/".format(
    datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(save_file_path, exist_ok=True)

# Preprocessing
data = preprocess.load_data()
(X_train, y_train, X_test, y_test, df_identifiers) = preprocess.preprocessing(data,
                                                                              encoder_method = ENCODER_METHOD,
                                                                              fill_na_method = FILL_NA_METHOD,
                                                                              datetime_treatment = DATEITME_TREATMENT)

# Learning

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


print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(y_test, yhat)))

# model_xgb_2 = xgb.Booster()
# model_xgb_2.load_model("model.json")

## Saving and Metrics
model.save_model(save_file_path + "/model.json")

description = {}
description["Date"] = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
description["User"] = os.getcwd().split("\\")[2]
description["Model"] = "Xgboost"
description["FILL_NA_METHOD"] = FILL_NA_METHOD
description["ENCODER_METHOD"] = ENCODER_METHOD
description["DATEITME_TREATMENT"] = DATEITME_TREATMENT

description["Metrics"] = "Root Mean Squared Error is {}".format(
    np.sqrt(mean_squared_error(y_test, yhat))
)
description[" \n Params Description"] = ""
description.update(model.get_params())
writing_description(description, save_file_path + "config.txt")
