from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None
        self.weights_ = None

    def use_weights(self, weights:np.ndarray):
        if self.fitted_:
            raise ValueError("Estimator cannot change its weights after calling ``fit``")
        self.weights_ = weights

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        assert len(y) == n_samples
        if self.weights_ is None:
            self.weights_ = np.ones(n_samples) / n_samples
        else:
            assert len(self.weights_) == n_samples

        thresholds = np.zeros(n_features)
        errs = np.zeros(n_features)
        signs = np.zeros(n_features)
        for j in range(n_features):
            pos_threshold, pos_err = self._find_threshold(X[:, j], y, 1)
            neg_threshold, neg_err = self._find_threshold(X[:, j], y, -1)

            if pos_err < neg_err:
                threshold, error, sign = pos_threshold, pos_err, 1
            else:
                threshold, error, sign = neg_threshold, neg_err, -1

            thresholds[j] = threshold
            errs[j] = error
            signs[j] = sign

        best_j = np.argmin(errs)
        self.threshold_ = thresholds[best_j]
        self.j_ = best_j
        self.sign_ = signs[best_j]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        n_samples, n_features = X.shape
        prediction = np.ones(n_samples)
        values = X[:, self.j_]
        positive_samples = values >= self.threshold_
        negative_samples = np.bitwise_not(positive_samples)
        prediction[negative_samples] = -1
        return prediction

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Comment1: It doesn't sound to me very efficient to scan like that all the values and test the error,
        # I would assume that something better can be done with sorting the values and using cumsum
        # to find the best threshold. Not sure exactly if it is much better
        #
        # Comment2:
        # We can know for sure that the best assignment for this threshold is assignment of
        # the last element in the negative box (otherwise, we would be better to take one step to
        # the left and decrease the loss
        losses = np.zeros(len(values))
        for m, value in enumerate(values):
            positive_samples = values >= value
            negative_samples = np.bitwise_not(positive_samples)

            pos_weights = self.weights_[positive_samples]
            neg_weights = self.weights_[negative_samples]

            loss_pos = np.sum(pos_weights[sign != labels[positive_samples]])
            loss_neg = np.sum(neg_weights[-sign != labels[negative_samples]])
            loss_pos = 0 if np.isnan(loss_pos) else loss_pos
            loss_neg = 0 if np.isnan(loss_neg) else loss_neg
            total_loss = loss_pos + loss_neg
            losses[m] = total_loss

        best_split = np.argmin(losses)
        return values[best_split], losses[best_split]

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
        y_predict = self.predict(X)
        return misclassification_error(y, y_predict)
