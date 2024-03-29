from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from scripts import random_state

__logistic_model = LogisticRegression(class_weight='balanced', random_state=random_state, n_jobs=-1)

__logistic_param_grid  = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
                        'penalty': ['l1', 'l2'], 
                        'solver': ['liblinear', 'saga']
                        }

__rf_model = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)

__rf_param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 5, stop = 15, num = 10)],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(0, 10, num = 10)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }


param_grid = {
    'LogisticRegression' : {'model' : __logistic_model,
                            'param_grid': __logistic_param_grid},
    'RandomForestClassifier':{'model' : __rf_model,
                            'param_grid': __rf_param_grid}
                            }
