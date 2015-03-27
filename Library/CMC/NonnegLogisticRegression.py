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
    for i in xrange(0, len(data) - batch_size, batch_size):
        yield data[i:(i+batch_size)]


def SetPositive(w):
    I = np.identity(9)
    I3 = np.hstack([I, I, I])
    I3 = 2*I3 - 1
    res = w*I3
    res[res < 0] = 0
    return res*I3

def _sgd(X, Y, alpha, max_epoch = 10, batch_size = 128, 
        validation = .10, random_state = None, eps = 1e-7):
    """
    Multinomial Classification
    """
    np.random.seed(random_state)
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    if type(validation) is float:
        assert 0 < validation < 1, "validation %g must be in (0, 1) %validation"
        cut = int(validation*len(Y))
        valid = idx[:cut]; train = idx[cut:]
    elif type(validation) is numpy.ndarray:
        valid = validation
        train = np.array([i for i in idx if i not in valid])
    else:
        raise Exception("Validation must be either a float for percentage,\
                or an array of index")
    logger.info("Train Size: %d, Valid Size: %d", len(train), len(valid))
    w_old = np.random.randn(Y.shape[1], X.shape[1])
    w_old = abs(w_old)/np.max(abs(w_old))
    w = np.random.randn(Y.shape[1], X.shape[1])
    w = abs(w)/np.max(abs(w))
    _, grad_old, _ = _multinomial_loss_grad(w_old, X, Y, alpha)
    logger.info("%10s | %10s", "Train Loss", "Valid Loss")
    for i in xrange(max_epoch):
        np.random.shuffle(train)
        for batch in _batch_gen(train, batch_size):
            _, grad, _ = _multinomial_loss_grad(w, X[batch], Y[batch], alpha)
            dgrad = grad - grad_old
            t = np.sum(dgrad*(w - w_old))/np.sum(dgrad**2) + eps
            w_old, grad_old = w, grad
            w = w_old - t*grad_old
            grad_old = grad
            w = SetPositive(w)
            #if loss < eps: break
            train_loss, _, _ = _multinomial_loss(w, X[train], Y[train], alpha)
            valid_loss, _, _ = _multinomial_loss(w, X[valid], Y[valid], alpha)
            logger.info("%10g | %10g", train_loss/len(train), 
                                       valid_loss/len(valid))
    return w


def _sgd2(X, Y, alpha, step_size = .001, max_epoch = 10, batch_size = 128, 
          validation = .1, random_state = None):
    np.random.seed(random_state)
    w = np.random.randn(Y.shape[1], X.shape[1])
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    if type(validation) is float:
        assert 0 < validation < 1, "validation %g must be in (0, 1) %validation"
        cut = int(validation*len(Y))
        valid = idx[:cut]; train = idx[cut:]
    elif type(validation) is numpy.ndarray:
        valid = validation
        train = np.array([i for i in idx if i not in valid])
    else:
        raise Exception("Validation must be either a float for percentage,\
                or an array of index")
    
    logger.info("%10s | %10s", "Train Loss", "Valid Loss")
    for i in xrange(max_epoch):
        np.random.shuffle(train)
        for batch in _batch_gen(train, batch_size):
            _, grad, _ = _multinomial_loss_grad(w, X[batch], Y[batch], alpha)
            w = w - step_size*grad
            w = SetPositive(w)
            loss, _, _ = _multinomial_loss(w, X, Y, alpha)
            train_loss, _, _ = _multinomial_loss(w, X[train], Y[train], alpha)
            valid_loss, _, _ = _multinomial_loss(w, X[valid], Y[valid], alpha)
            logger.info("%10g | %10g", train_loss/len(train),
                                       valid_loss/len(valid))
    return w

if False:
    np.random.seed(2)
    n_obs = 10000
    n_class = 3
    n_features = 100
    X = np.random.randn(n_obs, n_features)
    beta = np.random.binomial(1, .1, (n_features, n_class))
    log_odd = X.dot(beta)
    prob = np.exp(log_odd)/(1 + np.exp(log_odd))
    y = np.array([np.argmax(i) for i in prob])
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    res1 =  _sgd(X, y, .1, max_epoch = 20, validation = .1, 
        batch_size = 1000, random_state = 1) 
    res2 = _sgd2(X, y, .1, step_size = 1./len(X), max_epoch = 20,
        batch_size = 1000, random_state = 1)

    _, y, _ = LoadData(); del _
    X1, X1test = GetDataset('L1_btc_svc_mpc')
    y = lb.fit_transform(y)
    I = np.identity(9)
    O = np.zeros((9,9))
    I1 = np.hstack([I, O, O])
    _multinomial_loss(I1, X1, y, 0)[0]/len(y)

