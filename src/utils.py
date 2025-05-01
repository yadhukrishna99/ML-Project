import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, parameters):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = parameters[list(parameters.keys())[i]]

            gs = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=True, refit=True)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            model_train_score = r2_score(y_train, y_train_pred)

            model_test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = (model_test_score, gs.best_params_)
        
        return report

    except Exception as e:
        raise CustomException(e, sys)

