import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot, logsumexp, squared_norm
import ipdb


logger = logging.getLogger(__name__)


def _multinomial_loss(w, X, Y, alpha):
    sample_weight = np.ones(len(Y))
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

def _multinomial_loss_grad(w, X, Y, alpha):
    sample_weight = np.ones(len(Y)) 
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)))
    loss, p, w = _multinomial_loss(w, X, Y, alpha)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
                grad[:, -1] = diff.sum(axis=0)
    return loss, grad, p

def _batch_gen(data, batch_size):
    for i in xrange(0, len(y), batch_size):
        yield data[i:(i+batch_size)]

def _sgd(X, Y, alpha, n_epoch = 10, batch_size = 128, random_state = None):
    np.random.seed(random_state)
    w_old = np.random.randn(Y.shape[1], X.shape[1])
    w = np.random.randn(Y.shape[1], X.shape[1])
    _, grad_old, _ = _multinomial_loss_grad(w_old, X, Y, alpha)
    logger.info("%10s", "Train Loss")
    for i in xrange(n_epoch):
        idx = np.arange(len(Y))
        np.random.shuffle(idx)
        for batch in _batch_gen(idx, batch_size):
            loss, grad, _ = _multinomial_loss_grad(w, X[idx], Y[idx], alpha)
            dgrad = grad - grad_old
            t = np.sum(dgrad*(w - w_old))/np.sum(dgrad**2)
            w_old = w
            w = w_old - t*grad
            grad_old = grad
            w[w < 0] = 0
            logger.info("%10g", loss)
    return w


if __name__ == '__main__':
    np.random.seed(1)
    n_obs = 10000
    n_class = 3
    n_features = 10
    X = np.random.randn(n_obs, n_features)
    y = np.random.randint(low = 0,high = n_class, size = n_obs)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    w = np.random.randn(n_class,n_features)
    print _sgd(X, y, .1, n_epoch=10, random_state = 1) 

