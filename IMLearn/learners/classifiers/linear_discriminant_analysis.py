from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
        self.n_classes = 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        print(n_classes)
        assert np.all(np.isclose(self.classes_, np.arange(n_classes)))
        # classes_index_dict = {}
        # for i in range(n_classes):
        #     classes_index_dict[self.classes_[i]] = self.classes_[i]

        mus = np.zeros((n_classes, d))

        nks = np.zeros(n_classes)
        indexes = np.zeros(m, dtype=np.int32)
        for i in range(m):
            # class_index = classes_index_dict[y[i]]
            class_index = y[i]
            nks[class_index] += 1
            mus[class_index, :] += X[i, :]
            indexes[i] = class_index

        self.pi_ = nks / m
        self.mu_ = mus / nks[:, np.newaxis]

        centered = X - self.mu_[indexes]
        self.cov_ = 1/m * centered.transpose() @ centered
        assert self.cov_.shape == (d, d)
        assert self.mu_.shape == (n_classes, d)
        self._cov_inv = np.linalg.inv(self.cov_)
        self.n_classes = n_classes

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
        mu_t = self.mu_.transpose()
        A = self._cov_inv @ self.mu_.transpose()
        print("pis", self.pi_.shape, "mus", self.mu_.shape, mu_t * (self._cov_inv @ mu_t))
        B = np.log(self.pi_) - 1 / 2 * np.sum(mu_t * (self._cov_inv @ mu_t), axis=0)
        print("A", A.shape, X.shape, 'Bshape', B.shape, self._cov_inv.shape)
        ys = A.transpose() @ X.transpose() + B[:, np.newaxis]
        ys = ys.transpose()
        assert ys.shape == (m, len(self.classes_))
        return ys

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
