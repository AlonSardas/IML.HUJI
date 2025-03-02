from __future__ import annotations

from typing import NoReturn

import numpy as np

from ...base import BaseEstimator
from ...metrics import loss_functions


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """

        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        X = self._fix_intercept(X)
        n_samples, n_features = X.shape
        assert n_samples >= n_features

        U, S, Vt = np.linalg.svd(X)
        S_lambda = S / (S ** 2 + self.lam_)
        S_lam_mat = np.zeros((n_features, n_samples))
        assert len(S_lambda) == n_features
        S_lam_mat[:n_features, :n_features] = np.diag(S_lambda)
        print(Vt.transpose().shape, S_lam_mat.shape, U.transpose().shape)
        mat = Vt.transpose() @ S_lam_mat @ U.transpose()
        w = mat @ y
        assert len(w) == n_features
        self.coefs_ = w

    def _fix_intercept(self, X):
        if self.include_intercept_:
            if X.ndim == 1:
                X = np.array([X]).transpose()
            m, d = X.shape
            X = np.append(X, np.ones((m, 1)), axis=1)
        return X

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        X = self._fix_intercept(X)
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_predict = self.predict(X)
        return loss_functions.mean_square_error(y_predict, y)
