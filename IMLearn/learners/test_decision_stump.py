import numpy as np

from IMLearn.learners.classifiers import DecisionStump


def test_single_feature():
    X = np.array([[1, 6, 2, 9, 10, 5]]).transpose()
    y = np.array([1, -1, 1, -1, -1, 1])
    fitter = DecisionStump()
    fitter.fit(X, y)
    print(fitter.sign_, fitter.j_, fitter.threshold_)
    assert fitter.threshold_ == 6
    assert fitter.sign_ == -1


def test_no_realizable():
    X = np.array([[1, 6, 2, 9, 10, 5]]).transpose()
    y = np.array([-1, -1, +1, -1, -1, 1])
    fitter = DecisionStump()
    fitter.fit(X, y)
    assert fitter.threshold_ == 6
    assert fitter.sign_ == -1


def test_weights():
    X = np.array([[1, 6, 2, 9, 10, 5]]).transpose()
    y = np.array([-1, -1, +1, -1, -1, 1])
    Ws = np.array([1, 0, 0, 0, 0, 0])
    fitter1 = DecisionStump()
    fitter1.fit(X, y)
    print(fitter1.sign_, fitter1.j_, fitter1.threshold_)
    fitter2 = DecisionStump()
    fitter2.use_weights(Ws)
    fitter2.fit(X, y)
    print(fitter2.sign_, fitter2.j_, fitter2.threshold_)
    assert fitter2.threshold_ == 1
    assert fitter2.sign_ == -1


def main():
    # test_single_feature()
    test_weights()
    # test_no_realizable()


if __name__ == '__main__':
    main()
