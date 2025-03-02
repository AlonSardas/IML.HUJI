from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None
        self.n_classes = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_classes = n_classes
        assert np.all(np.isclose(self.classes_, np.arange(n_classes)))

        mus = np.zeros((n_classes, d))

        nks = np.zeros(n_classes)
        indexes = np.zeros(m, dtype=np.int32)
        for i in range(m):
            class_index = y[i]
            nks[class_index] += 1
            mus[class_index, :] += X[i, :]

            indexes[i] = class_index
        self.pi_ = nks / m
        self.mu_ = mus / nks[:, np.newaxis]

        sigmas = np.zeros((n_classes, d))
        centered = X - self.mu_[indexes]
        squared = centered ** 2
        for i in range(m):
            class_index = y[i]
            sigmas[class_index, :] += squared[i, :]

        self.vars_ = sigmas / nks[:, np.newaxis]

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
        ys = self.likelihood(X)
        return np.argmax(ys, axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m, d = X.shape
        mu = self.mu_
        mu_t = self.mu_.transpose()

        # The axes are k, i, j
        C = X[np.newaxis, :, :] - mu[:, np.newaxis, :]
        assert C.shape == (self.n_classes, X.shape[0], X.shape[1])
        C_squared = C ** 2
        exp = np.exp(-1/(2*self.vars_[:, np.newaxis, :]) * C_squared)
        factor = self.pi_[:, np.newaxis, np.newaxis] * (2*np.pi*self.vars_[:, np.newaxis, :]) ** (-1/2)
        ans = factor * exp
        ans = np.prod(ans, 2)
        return ans.transpose()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        misclassification_error(y, self.predict(X))
