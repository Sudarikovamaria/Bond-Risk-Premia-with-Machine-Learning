"""PCR, PLS и штрафные регрессии (панели A и B таблиц 1-2)."""

from functools import partial

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

RIDGE_GRID = np.logspace(-4, 4, 25)


def penalty_grid(X_train, y_train, l1_ratio=1.0, n_alphas=25, eps=1e-3):
    """Сетка штрафов относительно данных: от значения, при котором все
    коэффициенты равны нулю, и на три порядка вниз.

    Абсолютная сетка здесь не годится. Решение ridge линейно по цели, поэтому его
    R² не зависит от того, заданы доходности в процентах или в долях. У lasso и
    elastic net штраф не однороден относительно квадратичного члена, и нужный
    alpha масштабируется вместе с целью: при переходе с долей на проценты он
    растёт в сто раз и выходит за фиксированную сетку.
    """
    std = X_train.std(0)
    std[std == 0] = 1.0
    X_scaled = (X_train - X_train.mean(0)) / std
    centered = y_train - y_train.mean()
    alpha_max = np.max(np.abs(X_scaled.T @ centered)) / (len(centered) * l1_ratio)
    return np.logspace(np.log10(alpha_max), np.log10(alpha_max * eps), n_alphas)


LASSO_GRID = partial(penalty_grid, l1_ratio=1.0)
ENET_GRID = partial(penalty_grid, l1_ratio=0.5)


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def pcr(X_train, y_train, X_test, n_components, squared=False):
    X_train, X_test = standardize(X_train, X_test)
    pca = PCA(n_components=n_components).fit(X_train)
    f_train, f_test = pca.transform(X_train), pca.transform(X_test)
    if squared:
        f_train = np.hstack([f_train, f_train**2])
        f_test = np.hstack([f_test, f_test**2])
    return LinearRegression().fit(f_train, y_train).predict(f_test)


def pls(X_train, y_train, X_test, n_components):
    X_train, X_test = standardize(X_train, X_test)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train, y_train)
    return model.predict(X_test).ravel()


def ridge(X_train, y_train, X_test, alpha):
    X_train, X_test = standardize(X_train, X_test)
    return Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)


def lasso(X_train, y_train, X_test, alpha):
    X_train, X_test = standardize(X_train, X_test)
    model = Lasso(alpha=alpha, max_iter=50_000)
    return model.fit(X_train, y_train).predict(X_test)


def elastic_net(X_train, y_train, X_test, alpha, l1_ratio=0.5):
    X_train, X_test = standardize(X_train, X_test)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50_000)
    return model.fit(X_train, y_train).predict(X_test)
