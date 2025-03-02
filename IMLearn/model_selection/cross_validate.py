from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # To create cv folds with equal size, we first shuffle the data and then take
    # batches of approximately the same size
    n_samples, n_features = X.shape
    assert len(y) == n_samples
    p = np.random.permutation(n_samples)
    X = X[p, :]
    y = y[p]

    test_fold_mask = np.zeros(n_samples, dtype=bool)
    train_scores = np.zeros(cv)
    validation_scores = np.zeros(cv)
    for i in range(cv):
        test_fold_mask[:] = False
        test_fold_mask[int(i * n_samples / cv): int((i + 1) * n_samples / cv)] = True
        train_X = X[~test_fold_mask]
        train_y = y[~test_fold_mask]
        test_X = X[test_fold_mask]
        test_y = y[test_fold_mask]

        estimator.fit(train_X, train_y)
        predict_y = estimator.predict(train_X)
        train_scores[i] = scoring(predict_y, train_y)

        predict_y = estimator.predict(test_X)
        validation_scores[i] = scoring(predict_y, test_y)

    return np.mean(train_scores), np.mean(validation_scores)
