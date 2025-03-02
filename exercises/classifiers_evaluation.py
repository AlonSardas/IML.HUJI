import os

import matplotlib.pyplot as plt

from IMLearn import BaseEstimator
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        filepath = os.path.join('..', 'datasets', f)
        X, y = load_dataset(filepath)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(fitter:Perceptron, xxxx, yyy):
            losses.append(fitter._loss(X, y))

        fitter = Perceptron(callback=callback)
        fitter.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig, ax = plt.subplots()
        ax.plot(losses, '.')
        plt.show()


def get_ellipse(ax, mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return ax.plot(mu[0] + xs, mu[1] + ys, c="k")[0]


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        filepath = os.path.join('..', 'datasets', f)
        X, y = load_dataset(filepath)

        # Fit models and predict over training set
        lda_fitter = LDA()
        lda_fitter.fit(X, y)

        naive_fitter = GaussianNaiveBayes()
        naive_fitter.fit(X, y)
        print(lda_fitter.mu_)
        print(naive_fitter.mu_)
        print(naive_fitter.vars_)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y)

        fig, axes = plt.subplots(1, 2)
        def plot(ax, fitter:BaseEstimator, name:str):
            predicted_ys = fitter.predict(X)
            acc = accuracy(y, predicted_ys)
            ax.set_title(f"{name}, accuracy={acc:.3f}")
            ax.scatter(X[:, 0], X[:, 1], c=predicted_ys)
            wrong_X = X[y != predicted_ys, :]
            ax.scatter(wrong_X[:, 0], wrong_X[:, 1], c='r', marker='x', s=20)

        plot(axes[0], naive_fitter, 'Naive Gaussian')
        mu = naive_fitter.mu_
        axes[0].scatter(mu[:, 0], mu[:, 1], c='k', marker='x', s=30)
        plot(axes[1], lda_fitter, 'LDA')
        mu = lda_fitter.mu_
        axes[1].scatter(mu[:, 0], mu[:, 1], c='k', marker='x', s=30)

        ax = axes[0]
        for k in range(naive_fitter.n_classes):
            vars = naive_fitter.vars_[k, :]
            cov = np.diag(vars)
            get_ellipse(ax, naive_fitter.mu_[k, :], cov)
        ax = axes[1]
        for k in range(lda_fitter.n_classes):
            get_ellipse(ax, lda_fitter.mu_[k, :], lda_fitter.cov_)

        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
