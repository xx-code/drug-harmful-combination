import numpy as np
import pandas as pd
import xgboost

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from modules.utils import load_dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, roc_auc_score, roc_curve, recall_score

from tqdm import tqdm



def test(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred,)
    roc_plot = roc_curve(y_true, y_pred)
    error = 1 - accuracy

    scores_ensemble = {
        'precision': precision,
        'error': error,
        'f1-score': f1,
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy,
        'roc_auc_score': roc_auc
    }

    return matrix, scores_ensemble, roc_plot