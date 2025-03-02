import matplotlib.pyplot as plt
import numpy as np

from IMLearn.learners.regressors import RidgeRegression


def test_regression():
    reg = RidgeRegression(lam=1)
    X = np.array([[1, 4, 6, 2, 10]]).transpose()
    Y = np.array([1, 7, 5, 2, -2])

    reg.fit(X, Y)
    print(reg.coefs_)
    print(reg.predict(X))

    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    xs = np.linspace(X.min(), X.max(), 100)
    ys = reg.predict(np.array([xs]).transpose())

    print(f"loss={reg.loss(X, Y)}")

    ax.plot(xs, ys, '--')
    plt.show()


def main():
    test_regression()
    # test_SVD()


if __name__ == '__main__':
    main()