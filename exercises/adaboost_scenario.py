from typing import Tuple

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from utils import *


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    fitter = AdaBoost(DecisionStump, n_learners)
    fitter.fit(train_X, train_y)

    test_losses = np.zeros(n_learners)
    train_losses = np.zeros(n_learners)
    # Again, it would have been smarter to use cumsum to sum over the partial loss per learner...
    Ts = np.arange(n_learners)
    for t in Ts:
        test_losses[t] = fitter.partial_loss(test_X, test_y, t + 1)
        train_losses[t] = fitter.partial_loss(train_X, train_y, t + 1)
    ax: Axes
    fig, ax = plt.subplots()
    ax.plot(Ts, test_losses, '-', label='test loss')
    ax.plot(Ts, train_losses, '-', label='train loss')
    ax.legend()
    ax.set_xlabel('N learners')
    ax.set_ylabel('partial loss')

    # plt.show()

    # Question 2: Plotting decision surfaces
    # T = [5, 50, 100, 250]
    # T = [5, 10, 30]
    T = [5, 10, 30, 50]
    fig, axes = plt.subplots(ncols=len(T))
    for i, t in enumerate(T):
        ax = axes[i]
        ax.set_title(f"T={t}")
        xs = train_X[:, 0]
        ys = train_X[:, 1]
        ax.scatter(xs, ys, c=train_y)

        lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
            [-.1, .1])
        xrange = lims[0]
        yrange = lims[1]
        pred = lambda data: fitter.partial_predict(data, t)
        plot_decision_boundaries(ax, xrange, yrange, pred)

    # plt.show()

    use_plotly = False
    if use_plotly:
        lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
            [-.1, .1])
        fig = go.Figure()
        for t in T:
            boxes = decision_surface(lambda Xs: fitter.partial_predict(Xs, t), lims[0], lims[1])
            fig.add_trace(boxes)
            dots = decision_surface(lambda Xs: fitter.partial_predict(Xs, t), lims[0], lims[1], dotted=1)
            fig.add_trace(dots)
        fig.write_image("fig1.png")
        # fig.show()

    # Question 3: Decision surface of best performing ensemble
    fig, ax = plt.subplots()
    best_t = np.argmin(test_losses)
    ax.set_title(f"best T={best_t}, accuracy={test_losses[best_t]}")
    xs = test_X[:, 0]
    ys = test_X[:, 1]
    ax.scatter(xs, ys, c=test_y)

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    pred = lambda data: fitter.partial_predict(data, best_t)
    plot_decision_boundaries(ax, lims[0], lims[1], pred)
    # plt.show()

    # Question 4: Decision surface with weighted samples
    pos = train_y == 1
    neg = train_y == -1
    D = fitter.D_
    D = D / np.max(D) * 10

    fig, ax = plt.subplots()
    xs = train_X[pos, 0]
    ys = train_X[pos, 1]
    D_pos = D[pos]
    for x, y, d in zip(xs, ys, D_pos):
        ax.scatter(x, y, c='b', s=d)
    xs = train_X[neg, 0]
    ys = train_X[neg, 1]
    D_neg = D[neg]
    for x, y, d in zip(xs, ys, D_neg):
        ax.scatter(x, y, c='r', s=d)
    plt.show()


def main():
    # fit_and_evaluate_adaboost(0, n_learners=30, train_size=1000, test_size=100)
    # fit_and_evaluate_adaboost(0, n_learners=50, train_size=1500, test_size=300)
    # fit_and_evaluate_adaboost(0, n_learners=50, train_size=100, test_size=10)
    fit_and_evaluate_adaboost(0.4, n_learners=50, train_size=1500, test_size=300)


if __name__ == '__main__':
    np.random.seed(0)
    main()
