import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot, logsumexp, squared_norm
import ipdb
from scipy import optimize
import ipdb
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

logger = logging.getLogger(__name__)

def _multinomial_loss(w, X, Y, alpha, sample_weight):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w

def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)))
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def logistic_regression_path(X, y,  Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             copy=True, dual=False, penalty='l2',
                             intercept_scaling=1.):
    if type(Cs) is int:
        Cs = np.logspace(-4, 4, Cs)
    #X = check_array(X, accept_sparse='csr', dtype=np.float64) 
    #y = check_array(y, ensure_2d=False, copy=copy, dtype=None)
    _, n_features = X.shape
    assert len(X) == len(y), "X and y must have the same length"
    
    classes = np.unique(y)
    lbin = LabelBinarizer()
    Y_bin = lbin.fit_transform(y)
    w0 = np.zeros((Y_bin.shape[1], n_features + int(fit_intercept)),
                                  order='F')
    w0 = w0.ravel()
    target = Y_bin
    n_vectors = classes.size

    sample_weight = np.ones(X.shape[0])
    func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
    coefs = list()
    for C in Cs:
        w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol,
                    maxiter = max_iter
                    )
        try:
            multi_w0 = np.reshape(w0, (classes.size, -1))
        except ValueError:
            ipdb.set_trace()
        coefs.append(multi_w0)
    return coefs, np.array(Cs)

class LogisticRegression2(BaseEstimator):
    def __init__(self, tol = 1e-4, C = 1.0, fit_intercept = True, 
            intercept_scaling = 1, random_state = None, max_iter = 100,
            verbose = 0):
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
    def fit(self, X, y):
        n_classes = len(np.unique(y))
        coef_, _ = logistic_regression_path(
                X, y, Cs = [self.C], 
                fit_intercept = self.fit_intercept, tol = self.tol, 
                verbose = self.verbose, max_iter = self.max_iter)
        coef_ = coef_[0]
        if self.fit_intercept:
            self.intercept_ = coef_[:, -1]
            self.coef_ = coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)
            self.coef_ = coef_
        
        return self
    def predict_proba(self, X):
        p = safe_sparse_dot(X, self.coef_.T)
        p += self.intercept_
        p -= logsumexp(p, axis=1)[:, np.newaxis]
        return np.exp(p, p)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis = 1)


if __name__ == '__main__':
    n = 1000; p = 10; k = 3
    X = np.random.randn(n, p)
    beta = np.random.binomial(1, .5, (p, k))
    log_odd = X.dot(beta)
    prob = np.exp(log_odd)/(1 + np.exp(log_odd))
    y = np.array([np.argmax(i) for i in prob])
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    w = randn(k,p)
    cut = n/2
    train = np.arange(cut); valid = np.arange(cut,n)
    cl1 = LogisticRegression()
    cl2 = LogisticRegression2()
    cl1.fit(X[train], y[train])
    cl2.fit(X[train], y[train])
    prob1 = cl1.predict_proba(X[valid])
    prob2 = cl2.predict_proba(X[valid])
    print log_loss(y[valid], prob1)
    print log_loss(y[valid], prob2)
    yhat1 = cl1.predict(X[valid])
    yhat2 = cl2.predict(X[valid])
    print accuracy_score(y[valid], yhat1)
    print accuracy_score(y[valid], yhat2)
