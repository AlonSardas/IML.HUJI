import numpy as np

from IMLearn.desent_methods import GradientDescent, FixedLR
from IMLearn.desent_methods.modules import LogisticModule
from IMLearn.learners.classifiers import LogisticRegression
from exercises import gradient_descent_investigation
from exercises.gradient_descent_investigation import load_data, get_gd_state_recorder_callback, plot_descent_path




def test_logistic_module_no_intercept():
    X_train, y_train, X_test, y_test = load_data(train_portion=0.8)
    X_train = X_train[["tobacco", "obesity"]]
    # X_train = X_train["obesity"]
    callback, values, weights = get_gd_state_recorder_callback()
    solver = GradientDescent(FixedLR(1e-2), max_iter=15000, callback=callback)
    fitter = LogisticRegression(include_intercept=False, solver=solver)
    fitter.fit(X_train, y_train)
    # X_train_wifitter._fix_intercept(X_train)
    fig = plot_descent_path(LogisticModule, np.array(weights),
                            module_args=dict(X=X_train, y=y_train))
    fig.show()


def test_logistic_module_with_intercept():
    X_train, y_train, X_test, y_test = load_data(train_portion=0.8)
    X_train = X_train[["obesity"]]
    # X_train = X_train["obesity"]
    callback, values, weights = get_gd_state_recorder_callback()
    solver = GradientDescent(FixedLR(1e-2), max_iter=15000, callback=callback)
    fitter = LogisticRegression(include_intercept=True, solver=solver)
    fitter.fit(X_train, y_train)
    X_train_with_intercept = fitter._fix_intercept(X_train)

    fig = plot_descent_path(LogisticModule, np.array(weights), xrange=(-100, 100),
                            module_args=dict(X=X_train_with_intercept, y=y_train))
    fig.show()


def main():
    pass
    # test_logistic_module_with_generated_data()


if __name__ == '__main__':
    main()
