import matplotlib.pyplot as plt

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = X.drop(["id", "lat", "yr_renovated", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)
    df = df[df['price'] > 0]
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for column in df:
        fig, ax = plt.subplots()
        X_data = df[column].values
        Y_data = y.values
        ax.scatter(X_data, Y_data)

        pearson = np.mean((X_data - np.mean(X_data)) * (Y_data - np.mean(Y_data))) / (np.var(X_data) * np.var(Y_data)) ** 0.5
        ax.set_title(f"{column}, pearson:{pearson:.5f}")

    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 2 - Preprocessing of housing prices dataset
    df = preprocess_data(df)

    # Question 1 - split data into train and test sets
    prices = df['price']
    df = df.drop(columns=['price'])
    X = df
    y = prices

    # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(df, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    X = X.loc[:, ['sqft_above', 'sqft_living', 'floors']]
    size = 3000
    X = X.loc[:size, :]  # this is too much data for my computer
    y = y.loc[:size]
    print(X.shape)
    X = X.loc[:, ['sqft_above']]
    ps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    repeats = 10
    MSEs = np.zeros((len(ps), repeats))
    for i, p in enumerate(ps):
        for j in range(repeats):
            train_X, train_y, test_X, test_y = split_train_test(X, y, train_proportion=p)
            print(train_X.shape, test_X.shape)
            fit = LinearRegression()
            fit.fit_predict(train_X.values, train_y.values)
            MSEs[i, j] = fit.loss(test_X.values, test_y.values)

            if False and j == 0:
                fig, ax = plt.subplots()
                ax.scatter(train_X.values, train_y.values)
                ax.scatter(test_X.values, test_y.values)
                xs = np.linspace(train_X.values.min(), train_X.values.max())
                ys = fit.predict(xs)
                ax.plot(xs, ys, '--')
                ax.set_title(f'p={p}')

    fig, ax = plt.subplots()
    ax.plot(ps*100, MSEs, '.')
    plt.show()

    fig, ax = plt.subplots()
    MSE = MSEs.mean(axis=1)
    stds = MSEs.std(axis=1)
    ax.plot(ps*100, MSE, '.')
    ax.plot(ps*100, MSE+2*stds, '.')
    ax.plot(ps*100, MSE-2*stds, '.')
    ax.set_xlabel('percentage')
    ax.set_ylabel('MSE')
    plt.show()
