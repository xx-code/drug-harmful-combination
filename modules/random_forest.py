import numpy as np

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from utils import load_dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel



def train(X, y, criterions=['gini'], n_tree=100, step_n_tree=50):

    list_n_tree = np.arange(50, n_tree, step_n_tree+1, dtype=np.int64)
    
    hyperparameters = [(tree, criterion) for tree in list_n_tree for criterion in criterions]

    clfs = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for tree, criterion in hyperparameters:
        clfs.append(
                    RandomForestClassifier(
                        n_estimators= tree,
                        criterion=criterion,
                        random_state=0 ))

    train_scores = []
    test_scores = []    
    
    for clf in clfs:
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    arg_best_score = np.argmax(test_scores)

    clf_final = {
                    'classifier': clfs[arg_best_score],
                    'score_train': train_scores[arg_best_score],
                    'score_test': test_scores[arg_best_score],
                    'n_tree': hyperparameters[arg_best_score][0],
                    'criterion': hyperparameters[arg_best_score][1]
                }
    
    history = []

    for clf, hyperparameter, train_score, test_score in zip(clfs, hyperparameters, train_scores, test_scores):
        history.append(
            {
                'score_train': train_score,
                'score_test': test_score,
                'n_tree': hyperparameter[0],
                'criterion': hyperparameter[1],
                'features_importances': clf.feature_importances_
            }
        )

    return history, clf_final


def test(clf, X, y):
    # error and precision
    precision = clf.score(X, y)
    error = 1 - precision

    # other
    y_pred = clf.predict(X)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    matrix = confusion_matrix(y, y_pred)

    scores_ensemble = {
        'precision': precision,
        'error': error,
        'f1-score': f1,
        'accuracy': accuracy
    }

    return matrix, scores_ensemble

def features_selection(n_tree, criterion, X, y):
    random_forest = RandomForestClassifier(n_estimators=n_tree, criterion=criterion)

    sfs = SequentialFeatureSelector(random_forest, n_features_to_select='auto')
    sfs.fit(X, y)
    
    list_feature_selected = sfs.get_support()

    new_X = sfs.transform(X)

    return list_feature_selected, new_X

def feature_selection_model(n_tree, criterion, X, y):
    random_forest = RandomForestClassifier(n_estimators=n_tree, criterion=criterion).fit(X, y)
    model = SelectFromModel(random_forest, prefit=True)

    new_X = model.transfer(X)
    list_feature_selected = model.get_support()

    return list_feature_selected, new_X