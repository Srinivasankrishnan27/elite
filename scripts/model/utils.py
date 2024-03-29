import numpy as np
import itertools
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scripts import cv, scoring_method_f1, scoring_method_roc_auc, thresholds
import matplotlib.pyplot as plt
import logging
import joblib
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

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



def train_model(X, Y, model_name, model, model_param_grid, save_model=True):
    """
    Train a machine learning model using GridSearchCV to find the best hyperparameters.

    Args:
        X (array-like): Features for training.
        Y (array-like): Target variable for training.
        model_name (str): Name of the model being trained.
        model (estimator): Instance of the machine learning model to train.
        model_param_grid (dict): Dictionary containing hyperparameter values to search.
        save_model (bool): Flag to save the model

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
    if save_model:
        joblib.dump(best_estimator, f'./model_repo/{model_name}.pkl')
        joblib.dump(best_estimator, f'./model_repo/grid_{model_name}.pkl')
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


def plot_roc_curve(y_true, predicted_prob, model_name, save_fig=True):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a binary classification model.

    Args:
        y_true (array-like): True labels.
        predicted_prob (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model.
        save_fig (bool, optional): Whether to save the plot as a PNG file. Default is True.

    Returns:
        None

    Raises:
        None

    Example:
        plot_roc_curve(Y_test, predicted_probabilities, 'Logistic Regression', save_fig=True)
    """
    fpr, tpr, _ = roc_curve(y_true, predicted_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_fig:
        plt.savefig(f'./model_repo/{model_name}_roc_curve.png')
    plt.show()


def get_report(y_true, predicted_prob, model_name):
    """
    Generate a report with evaluation metrics for various probability thresholds.

    Args:
        y_true (array-like): True labels.
        predicted_prob (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model.

    Returns:
        DataFrame: A pandas DataFrame containing evaluation metrics for each threshold.

    Raises:
        None

    Example:
        report_df = get_report(Y_test, predicted_probabilities, 'Logistic Regression')
    """

    metrics = []
    for threshold in thresholds:
        predictions_threshold = (predicted_prob[:, 1] > threshold).astype(int)
    
        # Evaluate model using predictions based on probability thresholds
        accuracy, precision, recall, f1, cm, class_accuracy, summary = evaluate_model(predictions_threshold, y_true)
    
        result_dict = {
            "Model_name": model_name,
            "Threshold" : threshold, 
            "Accuracy"  : accuracy, 
            "Precision" : precision, 
            "Recall"    : recall, 
            "F1 Score"  : f1, 
            "Class Accuracy": class_accuracy, 
            "summary" : summary,
            "cm" : cm
            }
        metrics.append(result_dict)
    return pd.DataFrame(metrics)


def plot_precision_recall_curve(y_true, predicted_prob, model_name):
    """
    Plot the Precision-Recall curve for a binary classification model.

    Args:
        y_true (array-like): True labels.
        predicted_prob (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model.

    Returns:
        None

    Raises:
        None

    Example:
        plot_precision_recall_curve(Y_test, predicted_probabilities, 'Logistic Regression')
    """
    precision, recall, thresholds = precision_recall_curve(y_true, predicted_prob[:, 1])

    hover_text = [f'Threshold: {threshold:.2f}<br>Recall: {recall_value:.2f}<br>Precision: {precision_value:.2f}'\
                   for threshold, recall_value, precision_value in zip(thresholds, recall, precision)]
    trace = go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve', hoverinfo='text', hovertext=hover_text)
    layout = go.Layout(title=f'Precision-Recall Curve - {model_name}', xaxis=dict(title='Recall'),yaxis=dict(title='Precision'))
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()