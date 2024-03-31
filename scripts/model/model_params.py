from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

from scripts import random_state

__default_param = {'class_weight': 'balanced', 'random_state':random_state, 'n_jobs':-1}

'''Logistic Regression'''
__logistic_model = LogisticRegression(**__default_param)

__logistic_param_grid  = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
                        'penalty': ['l1', 'l2'], 
                        'solver': ['liblinear', 'saga']
                        }

'''RandomForest Classifier'''
__rf_model = RandomForestClassifier(**__default_param)

__rf_param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 5, stop = 15, num = 10)],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(1, 10, num = 10)],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
    }

'''ExtraTreesClassifier'''
__tree_clf_model= ExtraTreesClassifier(**__default_param)

__tree_clf_param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 5, stop = 15, num = 10)],
    'max_features': ['log2', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(1, 10, num = 10)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }


'''MLPClassifer'''

__mlp_model = MLPClassifier(random_state=random_state,solver='sgd',early_stopping=True)
__mlp_param_grid = {
    'momentum': [0.1,0.05,0.2,0.3],
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']
    }



param_grid = {
    'LogisticRegression' : {'model' : __logistic_model, 'param_grid': __logistic_param_grid},
    'RandomForestClassifier':{'model' : __rf_model, 'param_grid': __rf_param_grid}, 
    'ExtraTreesClassifier' : {'model' : __tree_clf_model, 'param_grid': __tree_clf_param_grid},
    'MLPClassifier' : {'model' : __mlp_model, 'param_grid': __mlp_param_grid}
    }