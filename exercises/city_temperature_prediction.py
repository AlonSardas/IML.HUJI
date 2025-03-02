import matplotlib.pyplot as plt

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, dtype={'Month': 'int'}, parse_dates=['Date', 'Year', 'Month', 'Day'])
    df = df[df["Temp"] > -50]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df['Month'] = pd.to_numeric(df['Month'])
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = df[df["Country"] == "Israel"]
    Xs = israel_data["DayOfYear"].values
    Ys = israel_data["Temp"].values
    fig, ax = plt.subplots()
    ax.scatter(Xs, Ys)

    print(type(df["Month"][2]))

    stds = israel_data.groupby('Month').agg({"Temp": "std"})
    xs = stds.index
    print(xs)
    ys = stds["Temp"].values
    fig, ax = plt.subplots()
    ax.bar(xs, ys)
    plt.show()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
