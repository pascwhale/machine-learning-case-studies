# Utilities
# This file will contain useful functions that will be used across multiple notebooks.
# It will help clean up the rest of the notebooks and keep them as short and concise as possible.
import numpy as np
import matplotlib.pyplot as plt


## Train Multiple Estimators of the Same Type with Different Parameters
def train_estimators(X, y, estimator_type, param_name, param_vals, **kwargs):
    """
    
    Trains multiple instances of `estimator_type` on (X, y) by setting argument
    named `param_name` to each value in `param_vals`. Prints a message before
    training each instance. Returns the list of trained estimators.
    
    For example:
    
       >>> train_estimators(X, y, DecisionTreeClassifier, 'max_depth', [1, 5, 10],
                            splitter='random', random_state=0)
    
       Training DecisionTreeClassifier(max_depth=1, random_state=0, splitter='random')...
       Training DecisionTreeClassifier(max_depth=5, random_state=0, splitter='random')...
       Training DecisionTreeClassifier(max_depth=10, random_state=0, splitter='random')...

       [DecisionTreeClassifier(max_depth=1, random_state=0, splitter='random'),
        DecisionTreeClassifier(max_depth=5, random_state=0, splitter='random'),
        DecisionTreeClassifier(max_depth=10, random_state=0, splitter='random')] 
    """
    trained_estimators = []
    for param_val in param_vals:
        print(f'Training {estimator_type.__name__}({param_name}={param_val}, ', ', '.join([f'{key}={value.__repr__()}' for key, value in kwargs.items()]), ')...', sep='')
        trained_estimators.append(estimator_type(**{param_name: param_val, **kwargs}).fit(X, y))
    return trained_estimators



## Score Multiple Estimators
def score_estimators(X, y, estimators):
    """Scores each estimator on (X, y), returning a list of scores."""
    scores = []
    for estimator in estimators:
        scores.append(estimator.score(X,y))
    return scores


## Plot Estimator Scores
def plot_estimator_scores(estimators, param_name, param_vals, X_train, y_train, X_test, y_test, X_val, y_val):
    """
    Plots the training, validation, and testing scores of a list of estimators,
    where `param_name` and `param_vals` are the same as for `train_estimators`.
    The estimator with best validation score will be highlighted with an 'x'.
    """
    # Your implementation here. Use as many lines as you need.
    plt.figure()

    training_scores = score_estimators(X_train, y_train, estimators)
    validation_scores = score_estimators(X_val, y_val, estimators)
    test_scores = score_estimators(X_test, y_test, estimators)

    plt.title(f'{type(estimators[0]).__name__} Score vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('score')
    plt.plot(training_scores, '-go')
    plt.plot(validation_scores, '-ro')
    plt.plot(test_scores, '--k')

    max_validation_index = validation_scores.index(max(validation_scores))

    plt.plot(validation_scores, '-r', marker='x', markersize=10, markevery=[max_validation_index])
    plt.xticks(np.arange(len(param_vals)), param_vals)

    line_legend = plt.legend(['train', 'validate', 'test'], loc=2)
    plt.legend([f'train = {training_scores[max_validation_index]:.3f}',
                f'val = {validation_scores[max_validation_index]:.3f}',
                f'test = {test_scores[max_validation_index]:.3f}'],
                loc=4,
                handles=None,
                labelcolor=['g', 'r', 'k'])

    plt.gca().add_artist(line_legend)

