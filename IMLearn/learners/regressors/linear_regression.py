from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv

from ...metrics import loss_functions


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
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
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fix_intercept(self, X):
        if self.include_intercept_:
            if X.ndim == 1:
                X = np.array([X]).transpose()
            m, d = X.shape
            X = np.append(X, np.ones((m, 1)), axis=1)
        return X

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

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
        assert X.shape[0] == y.shape[0], "Shape mismatch"
        X = self._fix_intercept(X)

        # Now we need to calculate the pseudo-inverse...
        U, S, Vt = np.linalg.svd(X)
        eps = 1e-8
        S[S < eps] = 0
        S_inv = 1/S
        S_inv[S_inv == np.inf] = 0
        Sigma_inv = np.zeros(X.shape)
        i = np.arange(len(S))
        Sigma_inv[i, i] = S_inv

        Xd = U @ Sigma_inv @ Vt
        Xd = Xd.transpose()

        self.coefs_ = Xd @ y


    @staticmethod
    def _calc_svd(X:np.ndarray):
        m, n = X.shape
        should_switch = m > n
        if should_switch:
            X = X.transpose()
        m, n = X.shape
        sym = X.transpose() @ X
        D, V = np.linalg.eig(sym)
        indx = np.argsort(D)[::-1]
        D = D[indx]
        V = V[:, indx]
        Sigma = np.sqrt(D)
        eps = 1e-8
        Sigma[Sigma < eps] = 0
        left = X @ V
        U = left / Sigma
        U = U[:m, :m]
        Sigma = Sigma[:m]
        # Sigma = np.diag(Sigma)
        # Sigma = Sigma[:m, :n]

        if should_switch:
            print("Switch")
            T = V
            V = U.transpose()
            U = T.transpose()
            Sigma = Sigma.transpose()

        return U, Sigma, V.transpose()

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
