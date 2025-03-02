import matplotlib.pyplot as plt
import numpy as np

from IMLearn.learners import MultivariateGaussian
from gaussian_estimators import UnivariateGaussian


def test_fit():
    X = np.random.normal(10, 1, 1000)

    gauss_fit = UnivariateGaussian()
    gauss_fit.fit(X)
    print(f"{gauss_fit.mu_:.3f}, {gauss_fit.var_:.3f}")


def plot_fits():
    Ns = np.arange(10, 1001, 10)
    differences = np.zeros(len(Ns))

    mu = 10

    for i, N in enumerate(Ns):
        X = np.random.normal(mu, 1, N)
        gauss_fit = UnivariateGaussian()
        gauss_fit.fit(X)
        differences[i] = abs(gauss_fit.mu_ - mu)

    fig, ax = plt.subplots()
    ax.plot(Ns, differences, '.')
    ax.set_xlabel("N sample size")
    ax.set_ylabel(r"$ \Delta = |\mu - \mu^*| $")

    plt.show()


def plot_pdf():
    X = np.random.normal(10, 1, 1000)

    gauss_fit = UnivariateGaussian()
    gauss_fit.fit(X)
    pdfs = gauss_fit.pdf(X)

    fig, ax = plt.subplots()
    ax.plot(X, pdfs, '.')
    ax.set_xlabel("X")
    ax.set_ylabel(r"$ N pdf $")

    plt.show()


def test_multivariate_fit():
    mu = np.array([0, 0, 4, 0])
    Sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    N = 1000
    X = np.random.multivariate_normal(mu, Sigma, N)
    fit = MultivariateGaussian()
    fit.fit(X)


def plot_log_likelihood():
    mu = np.array([0, 0, 4, 0])
    Sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    N = 1000
    X = np.random.multivariate_normal(mu, Sigma, N)
    # fit = MultivariateGaussian()
    # fit.fit(X)

    f1s = np.linspace(-10, 10, 200)
    f3s = np.linspace(-10, 10, 200)
    logs = np.zeros((len(f3s), len(f1s)))
    for i, f1 in enumerate(f1s):
        for j, f3 in enumerate(f3s):
            mu_fs = np.array([f1, 0, f3, 0])
            logs[i, j] = MultivariateGaussian.log_likelihood(mu_fs, Sigma, X)

    amax = np.argmax(logs)
    i_max, j_max = np.unravel_index(amax, logs.shape)
    print(f"Maximum: f1:{f1s[i_max]:.3f}, f3:{f3s[j_max]:.3f}")


    fig, ax = plt.subplots()
    im = ax.imshow(logs, extent=[-10, 10, -10, 10])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$ \mu_3 $")
    ax.set_ylabel(r"$ \mu_1 $")

    plt.show()


def main():
    # plot_fits()
    # plot_pdf()
    # test_multivariate_fit()
    plot_log_likelihood()


if __name__ == '__main__':
    main()
