import numpy as np
import itertools
from data import GetDataset, LoadData
from sklearn.svm import SVC
from scipy.stats import uniform
from gl import BoostedTreesClassifier
from scipy.stats import randint
import cPickle as pickle
from sklearn.metrics import log_loss

X, _ = GetDataset('original')
_, y, _ = LoadData()

train = np.arange(1000, dtype = int)
valid = np.arange(1000, len(y))

if False:
    Cs = [.001, .01, .1, 1., 10.]
    gammas = [.001, .01, .1, 1., 10.]
    res = []; i = 0
    for (C, gamma) in itertools.product(Cs, gammas):
        print i, C, gamma; i += 1
        clf = SVC(C = C, gamma = gamma)
        clf.fit(X[train], y[train])
        res.append(clf.score(X[valid], y[valid]))

    res2 = []
    for i in xrange(len(res)*100):
        C     = 10**uniform(-3.5,5).rvs()
        gamma = 10**uniform(-3.5,5).rvs()
        print i, C, gamma
        clf = SVC(C = C, gamma = gamma)
        clf.fit(X[train], y[train])
        res2.append(clf.score(X[valid], y[valid]))

if True:
    grid_max_iterations   = [10, 20, 30]
    grid_step_size        = [.5, .7, .9]
    grid_max_depth        = [ 5,  7,  9]
    grid_row_subsample    = [.5, .7, .9]
    grid_column_subsample = [.5, .7, .9]
    iterator = itertools.product(grid_max_iterations,
            grid_step_size,
            grid_max_depth,
            grid_row_subsample,
            grid_column_subsample)

    res3 = []; i = 0
    for (mi, s, md, r, c) in iterator:
        print i, mi, s, md, r, c; i += 1
        clf = BoostedTreesClassifier(max_iterations = mi,
                step_size = s,
                max_depth = md,
                row_subsample = r,
                column_subsample = c)
        clf.fit(X[train], y[train])
        yhat = clf.predict_proba(X[valid])
        res3.append(log_loss(y[valid], yhat))
    pickle.dump(res3, open("BTCGridLL.pkl", 'w'))

np.random.seed(1)
if False:
    res4 = []
    for i in xrange(100*3**5):
        mi = randint(low =  5, high  = 36).rvs()
        s  = uniform(loc = .4, scale = .6).rvs()
        md = randint(low =  4, high  = 11).rvs()
        r  = uniform(loc = .4, scale = .6).rvs()
        c  = uniform(loc = .4, scale = .6).rvs()
        print i, mi, s, md, r, c
        clf = BoostedTreesClassifier(max_iterations = mi,
                step_size = s,
                max_depth = md,
                row_subsample = r,
                column_subsample = c)
        clf.fit(X[train], y[train])
        yhat = clf.predict_proba(X[valid])
        res4.append(log_loss(y[valid], yhat))
        pickle.dump(res4, open("BTCRandomizedLL.pkl", 'w'))

if False:
    res5 = []
    import sys
    sys.path.insert(0, '../Library/GPUCB')
    from gpucb              import *
    def f(mi, s, md, r, c):
        clf = BoostedTreesClassifier(max_iterations = mi,
                step_size = s,
                max_depth = md,
                row_subsample = r,
                column_subsample = c,
                verbose = 0)
        clf.fit(X[train], y[train])
        yhat = clf.predict_proba(X[valid])
        return -log_loss(y[valid], yhat)
    params_dist = {
            'mi': UniformInt(5, 36),
            's' : Uniform(.4, 1.),
            'md': UniformInt(4, 11),
            'r' : Uniform(.4, 1.),
            'c' : Uniform(.4, 1.),
            }
    for i in xrange(100):
        gp = GPUCBOpt(kernel = DoubleExponential, max_iter = 243,
                mu_prior = -1, sigma_prior = .20, sig = .005, 
                n_grid = 1000, time_budget = 3600*10, verbose = 1)
        gp.fit(func = f, params_dist = params_dist)
        res5.append(gp.y)
