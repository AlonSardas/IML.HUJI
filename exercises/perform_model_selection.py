from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso, lasso_path

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    dataset = datasets.load_diabetes()
    X = dataset.data
    y = dataset.target

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # train_X, train_y, test_X, test_y = split_train_test(X, y,
    #                                                     train_proportion=n_samples / len(y))
    train_X, train_y = X, y
    ridge_alphas, lasso_alphas = np.linspace(1e-5, 5 * 1e-2, num=n_evaluations), np.linspace(.001, 2, num=n_evaluations)
    ridge_scores, lasso_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    for i, (ra, la) in enumerate(zip(*[ridge_alphas, lasso_alphas])):
        ridge_scores[i] = cross_validate(RidgeRegression(ra), train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(Lasso(la, max_iter=5000), train_X, train_y, mean_square_error)

    def plot_score_vs_alpha(ax:Axes, alphas, scores, title):
        ax.set_title(title)
        ax.set_ylabel("loss")
        ax.set_xlabel(r"$\lambda$ regularization parameter")
        ax.plot(alphas, scores[:, 0], label='train score')
        ax.plot(alphas, scores[:, 1], label='validation score')
        ax.legend()

    fig, axes = plt.subplots(1, 2)
    plot_score_vs_alpha(axes[0], ridge_alphas, ridge_scores, "Ridge regularization")
    plot_score_vs_alpha(axes[1], lasso_alphas, lasso_scores, "Lasso regularization")

    plt.show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


def main():
    select_regularization_parameter()


if __name__ == '__main__':
    np.random.seed(0)
    main()
