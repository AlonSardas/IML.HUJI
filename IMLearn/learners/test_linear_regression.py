import matplotlib.pyplot as plt
import numpy as np

from IMLearn.learners.regressors import LinearRegression


def test_regression():
    reg = LinearRegression()
    X = np.array([[1, 4, 6, 2, 10]]).transpose()
    Y = np.array([1, 7, 5, 2, -2])

    print(X)
    print(X.shape)
    print(X.transpose() @ Y)


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


def test_SVD():
    X = np.array([[3,2,2],[2,3,-2]])
    U1, S1, Vt1 = LinearRegression._calc_svd(X)
    SS = np.zeros(X.shape)
    i = np.arange(len(S1))
    SS[i, i] = S1
    print(U1 @ SS @ Vt1)
    assert np.all(np.isclose(U1 @ SS @ Vt1, X))

    X = np.array([[3,0],[2,0]])
    U1, S1, Vt1 = LinearRegression._calc_svd(X)
    SS = np.zeros(X.shape)
    i = np.arange(len(S1))
    SS[i, i] = S1
    print(U1 @ SS @ Vt1)
    assert np.all(np.isclose(U1 @ SS @ Vt1, X))




def main():
    test_regression()
    # test_SVD()


if __name__ == '__main__':
    main()