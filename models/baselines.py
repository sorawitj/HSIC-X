import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.sandbox.regression.gmm import LinearIVGMM
from statsmodels.tools.tools import add_constant

from helpers.utils import radial


class PredPolyRidge(object):
    def __init__(self, degree, alpha=0.2, bias=False):
        """
        @param degree: the polynomial's degree
        @param alpha: the regularization parameter for Ridge
        @param bias: include a bias term or not
        """
        self.degree = degree
        self.bias = bias
        self.alpha = alpha
        if alpha == 0:
            self.reg = LinearRegression(fit_intercept=self.bias)
        else:
            self.reg = Ridge(fit_intercept=self.bias, alpha=self.alpha)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]

        return np.concatenate([X ** i for i in range(1, self.degree + 1)], 1)

    def fit(self, X, Y):
        # 2SLS
        self.reg.fit(self.make_features(X), Y)

    def predict(self, X_test):
        y_pred = self.reg.predict(self.make_features(X_test))

        return y_pred


class PredRadialRidge(object):

    def __init__(self, num_basis, data_limits=(-5, 5), alpha=0.2, bias=False):
        """
        @param num_basis: number of radial basis functions
        @param data_limits: min and max to create a grid for the radial basis functions
        @param alpha: the regularization parameter for Ridge
        @param bias: include a bias term or not
        """
        self.num_basis = num_basis
        self.data_limits = data_limits
        self.bias = bias
        self.alpha = alpha
        self.reg = Ridge(random_state=123, fit_intercept=False, alpha=self.alpha)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        phi = radial(X, num_basis=self.num_basis, data_limits=self.data_limits)
        if self.bias:
            phi = add_constant(phi)

        return phi

    def fit(self, X, Y):
        # 2SLS
        self.reg.fit(self.make_features(X), Y)

    def predict(self, X_test):
        y_pred = self.reg.predict(self.make_features(X_test))

        return y_pred


class Poly2SLS(object):
    def __init__(self, degree, bias=False):
        """
        @param degree: the polynomial's degree
        @param bias: include a bias term or not
        """
        self.degree = degree
        self.bias = bias

    def make_features(self, X, bias):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        if bias:
            start = 0
        else:
            start = 1

        return np.concatenate([X ** i for i in range(start, self.degree + 1)], 1)

    def fit(self, X, Y, Z):
        gmm = LinearIVGMM(Y, self.make_features(X, False), self.make_features(Z, False))
        self.gmm_res = gmm.fit()

    def predict(self, X_test):
        y_2sls = self.gmm_res.predict(self.make_features(X_test, False))

        return y_2sls


class Radial2SLS(object):
    def __init__(self, num_basis, data_limits=(-5, 5), bias=False):
        """
        @param num_basis: number of radial basis functions
        @param data_limits: min and max to create a grid for the radial basis functions
        @param bias: include a bias term or not
        """
        self.num_basis = num_basis
        self.data_limits = data_limits
        self.bias = bias

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        phi = radial(X, num_basis=self.num_basis, data_limits=self.data_limits)
        if self.bias:
            phi = add_constant(phi)

        return phi

    def fit(self, X, Y, Z):
        gmm = LinearIVGMM(Y, self.make_features(X), self.make_features(Z))
        self.gmm_res = gmm.fit()

    def predict(self, X_test):
        y_2sls = self.gmm_res.predict(self.make_features(X_test))

        return y_2sls


class Radial2SLSRidge(object):
    def __init__(self, num_basis, data_limits=(-5, 5), alpha=0.2, bias=False):
        """
        @param num_basis: number of radial basis functions
        @param data_limits: min and max to create a grid for the radial basis functions
        @param alpha: the regularization parameter for Ridge
        @param bias: include a bias term or not
        """
        self.num_basis = num_basis
        self.data_limits = data_limits
        self.alpha = alpha
        self.bias = bias
        self.reg1 = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=False, alpha=self.alpha))
        self.reg2 = Ridge(random_state=123, fit_intercept=False, alpha=self.alpha)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        phi = radial(X, num_basis=self.num_basis, data_limits=self.data_limits)
        if self.bias:
            phi = add_constant(phi)

        return phi

    def fit(self, X, Y, Z):
        # 2SLS
        self.reg1.fit(self.make_features(Z), self.make_features(X))
        X_hat = self.reg1.predict(self.make_features(Z))
        self.reg2.fit(X_hat, Y)

    def predict(self, X_test):
        y_2sls = self.reg2.predict(self.make_features(X_test))

        return y_2sls


class Poly2SLSRidge(object):
    def __init__(self, degree, alpha=0.2, bias=False):
        """
        @param degree: the polynomial's degree
        @param alpha: the regularization parameter for Ridge
        @param bias: include a bias term or not
        """
        self.degree = degree
        self.alpha = alpha
        self.bias = bias
        self.reg1 = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=False, alpha=self.alpha))
        self.reg2 = Ridge(random_state=123, fit_intercept=False, alpha=self.alpha)

    def make_features(self, X):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        if X.ndim == 1:
            X = X[:, np.newaxis]
        # add a column of one if bias is True
        if self.bias:
            start = 0
        else:
            start = 1

        return np.concatenate([X ** i for i in range(start, self.degree + 1)], 1)

    def fit(self, X, Y, Z):
        # 2SLS
        self.reg1.fit(self.make_features(Z), self.make_features(X))
        X_hat = self.reg1.predict(self.make_features(Z))
        self.reg2.fit(X_hat, Y)

    def predict(self, X_test):
        y_2sls = self.reg2.predict(self.make_features(X_test))

        return y_2sls
