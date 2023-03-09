import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import load_dataset


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
