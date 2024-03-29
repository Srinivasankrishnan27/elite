import numpy as np
import itertools
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scripts import cv, scoring_method_f1, scoring_method_roc_auc
import logging
import joblib


def feature_selection(X, Y, cat_cols, num_cols, clf, scoring=scoring_method_roc_auc, method='load'):
    '''Module to select features for the model
        Args:
            X: Pandas Dataframe, Independent variable
            Y: Pandas Dataframe, Dependent variable (Target)
            clf: estimator
            scoring: scorer callable object, default to roc_auc
            method: select/load
        Returns:
            selector : Fitted estimator object
            selected_cols: Selected columns as list
            
    '''
    if method == 'select':
        available_cols = X.columns
        logging.info(f'Number of input columns: {len(available_cols)}')
        logging.info(f'Feature selection started')
        selector = RFECV(estimator=clf, step=0.1, cv=cv, n_jobs=-1, scoring=scoring)
        selector = selector.fit(X, Y)
        feature_index = selector.support_.tolist()
        selected_cols = list(itertools.compress(available_cols, feature_index))
        logging.info(f'Number of columns selected: {len(selected_cols)}')
        dump_dict = {
            'selector' : selector,
            'selected_cols': selected_cols}
        joblib.dump(dump_dict, './model_repo/selected_cols.pkl')
    elif method == 'load':
        dump_dict = joblib.load('./model_repo/selected_cols.pkl')
        selector = dump_dict['selector']
        selected_cols = dump_dict['selected_cols']
    else:
        raise RuntimeError(f"Invalid method {method}")
    selected_num_cols = [ i for i in selected_cols if i in num_cols]
    selected_cat_cols = [ i for i in selected_cols if i in cat_cols]
    return selector, selected_cols, selected_num_cols, selected_cat_cols



def train_model(X, Y, model_name, model, model_param_grid):
    """
    Train a machine learning model using GridSearchCV to find the best hyperparameters.

    Args:
        X (array-like): Features for training.
        Y (array-like): Target variable for training.
        model_name (str): Name of the model being trained.
        model (estimator): Instance of the machine learning model to train.
        model_param_grid (dict): Dictionary containing hyperparameter values to search.

    Returns:
        tuple: A tuple containing the trained GridSearchCV object and the best estimator found.

    Raises:
        None

    Example:
        grid_search, best_estimator = train_model(X_train, Y_train, 'Logistic Regression',
                                                   LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10]})
    """
    logging.info(f'Model training started for : {model_name}')
    logging.info(f'Searching best params')
    grid_search = GridSearchCV(model, model_param_grid, cv=cv, scoring=scoring_method_f1, n_jobs=-1)
    logging.info(f'Param search completed')
    grid_search.fit(X, Y)
    best_params = grid_search.best_params_
    logging.info(f'Best params : {best_params}')
    best_estimator = grid_search.best_estimator_
    return grid_search, best_estimator



def evaluate_model(predictions, true_labels, target_names=['NORMAL','AFFLUENT']):
    """
    Evaluate a machine learning model using various classification metrics.

    Args:
        predictions (array-like): Predicted labels.
        true_labels (array-like): True labels.
        target_names (list): List of class labels.

    Returns:
        tuple: A tuple containing evaluation metrics including accuracy, precision, recall, F1-score,
               class-wise accuracy, and a summary report.

    Raises:
        None

    Example:
        accuracy, precision, recall, f1, class_accuracy, summary = evaluate_model(predictions, true_labels, target_names)
    """
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    summary = classification_report(true_labels, predictions, target_names=target_names, output_dict =True)
    cm = confusion_matrix(true_labels, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_accuracy = cm.diagonal()
    return accuracy, precision, recall, f1, cm, class_accuracy, summary