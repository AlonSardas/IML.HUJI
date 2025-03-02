import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, Optional

import sklearn.metrics
from matplotlib.axes import Axes

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from cross_validate import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5),
                      module_args:Optional[dict]=None) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    module_args = {} if module_args is None else module_args
    def predict_(w):
        return np.array([module(weights=wi).compute_output(**module_args) for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module_type in [L1, L2]:
        convergence_fig = go.Figure(layout=go.Layout(
            title=f'Convergence rate for {module_type.__name__}',
            xaxis=dict(title="t iteration"), yaxis=dict(title="loss")))
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            f = module_type(init.copy())
            gd = GradientDescent(FixedLR(eta), callback=callback)
            gd.fit(f, None, None)
            # fig = plot_descent_path(module_type, np.array(weights),
            #                         f"{module_type}, eta={eta}")
            # fig.show()
            convergence_fig.add_trace(go.Scatter(
                x=np.arange(len(values)), y=values, mode='lines', name=fr'$\eta={eta}$'))
        convergence_fig.show()

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    convergence_fig = go.Figure(layout=go.Layout(
        title=f'Convergence rate',
        xaxis=dict(title="t iteration"), yaxis=dict(title="loss")))
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        f = L1(init.copy())
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gd.fit(f, None, None)
        # fig = plot_descent_path(module_type, np.array(weights),
        #                         f"{module_type}, eta={eta}")
        # fig.show()
        convergence_fig.add_trace(go.Scatter(
            x=np.arange(len(values)), y=values, mode='lines', name=fr'$\gamma={gamma}$'))
    convergence_fig.show()

    # Plot descent path for gamma=0.95
    gamma = 0.95
    callback, values, weights = get_gd_state_recorder_callback()
    f = L1(init.copy())
    gd = GradientDescent(ExponentialLR(eta, gamma), out_type="best", callback=callback)
    best_weights = gd.fit(f, None, None)
    print(f"Best norm: {L1(best_weights).compute_output()}")
    fig = plot_descent_path(L1, np.array(weights),
                            f"L1 path, gamma={gamma}")
    fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data(train_portion=0.7)

    # Plotting convergence rate of logistic regression over SA heart disease data
    solver = GradientDescent()
    fitter = LogisticRegression(solver=solver)
    fitter.fit(X_train, y_train)
    score = fitter.predict_proba(X_test)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, score)
    print(score)
    print(thresholds)

    ax: Axes
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot((0, 1), (0, 1), '--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    best_index = np.argmax(tpr - fpr)
    fig, ax = plt.subplots()
    ax.plot(thresholds, tpr)
    ax.plot(thresholds, fpr)
    print(f"Best threshold: {thresholds[best_index]}")
    # plt.show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # Fitting regularized logistic regression, while choosing lambda using cross-validaiton
    # options = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    options = [0.001, 0.005, 0.01, 0.02, 0.1]
    # options = [0.001, 0.01, 0.1, 1]
    for penalty, lambdas in [("l1", options),
                             ("l2", options)]:
        # Running cross validation
        scores = np.zeros((len(lambdas), 2))
        for i, lam in enumerate(lambdas):
            gd = GradientDescent(learning_rate=FixedLR(1e-3), max_iter=20000)
            logistic = LogisticRegression(solver=gd, penalty=penalty, lam=lam, alpha=.5)
            scores[i] = cross_validate(estimator=logistic, X=X_train.values, y=y_train.values,
                                       scoring=misclassification_error)
            print(
                f'penalty is {penalty}, lambda is {lam}, train error is {scores[i, 0]}, test error is  {scores[i, 1]}')

        ax:Axes
        fig, ax = plt.subplots()
        ax.plot(lambdas, scores[:, 0], label="train score")
        ax.plot(lambdas, scores[:, 1], label="validation score")
        ax.legend()
        ax.set_xlabel(r'$\lambda $')
        ax.set_ylabel("error")


if __name__ == '__main__':
    np.random.seed(0)
    # test_logistic_module_no_intercept()
    # test_logistic_module_with_intercept()
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
