import matplotlib.pyplot as plt
import numpy as np

from IMLearn.learners.regressors import LinearRegression, PolynomialFitting


def test_fit():
    fit = PolynomialFitting(k=8)
    X = np.array([-1, 1, 4, 6, 2, 10, 15, 17])
    Y = np.array([5, 1, 7, 5, 2, -2, 10, 30])

    fit.fit(X, Y)
    print(fit.lin_reg.coefs_)

    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    xs = np.linspace(X.min(), X.max(), 100)
    ys = fit.predict(xs)

    # print(f"loss={reg.loss(X, Y)}")

    ax.plot(xs, ys, '--')
    plt.show()



def main():
    test_fit()


if __name__ == '__main__':
    main()