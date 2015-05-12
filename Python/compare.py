import numpy as np
import itertools
from scipy.stats import uniform
from scipy.stats import randint
import logging 

from ml import *


X, _ = GetDataset('original')
_, y, _ = LoadData()

from gl import BoostedTreesClassifier
clf = BoostedTreesClassifier(verbose = 0)
clf.fit(X[:10], y[:10])
np.random.seed(1)
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

## 3. Grid
if False:
    grid_max_iterations   = [10, 20, 30]
    grid_step_size        = [.5, .7, .9]
    grid_max_depth        = [ 5,  7,  9]
    grid_row_subsample    = [.5, .7, .9]
    grid_column_subsample = [.5, .7, .9]
    Grid = []; j = 0
    kcv = StratifiedKFold(y, n_folds = 62, random_state = 1)
    logger.info('Starting Search...\n')
    logger.info('   j|   i|  mi|   s|  md|   r|   c|')
    logger.info('____|____|____|____|____|____|____|')
    for train_idx, valid_idx in kcv:
        res3 = []; i = 0
        iterator = itertools.product(grid_max_iterations,
                grid_step_size,
                grid_max_depth,
                grid_row_subsample,
                grid_column_subsample)
        for (mi, s, md, r, c) in iterator:
            logger.info('%4d|%4d|%4d|%4.2g|%4d|%4.2g|%4.2g|',
                    j, i, mi, s, md, r, c); i += 1
            clf = BoostedTreesClassifier(max_iterations = mi,
                    step_size = s,
                    max_depth = md,
                    row_subsample = r,
                    column_subsample = c,
                    verbose = 0)
            clf.fit(X[valid_idx], y[valid_idx])
            #yhat = clf.predict_proba(X[train_idx])
            #res3.append(log_loss(y[train_idx], yhat))
            res3.append(clf.score(X[train_idx], y[train_idx]))
        Grid.append(res3)
        pickle.dump(Grid, open("BTCGridAccuracyFull.pkl", 'w'))
        j += 1

## 4. Rand
if True:
    Rand = []; j = 0
    kcv = StratifiedKFold(y, n_folds = 62, random_state = 1)
    logger.info('Starting Search...\n')
    logger.info('   j|   i|  mi|   s|  md|   r|   c|')
    logger.info('____|____|____|____|____|____|____|')
    for train_idx, valid_idx in kcv:
        if j < 37:
            j += 1
            continue
        res4 = []
        for i in xrange(3**5):
            mi = randint(low =  5, high  = 36).rvs()
            s  = uniform(loc = .4, scale = .6).rvs()
            md = randint(low =  4, high  = 11).rvs()
            r  = uniform(loc = .4, scale = .6).rvs()
            c  = uniform(loc = .4, scale = .6).rvs()
            logger.info('%4d|%4d|%4d|%4.2g|%4d|%4.2g|%4.2g|',
                    j, i, mi, s, md, r, c)
            clf = BoostedTreesClassifier(max_iterations = mi,
                    step_size = s,
                    max_depth = md,
                    row_subsample = r,
                    column_subsample = c,
                    verbose = 0)
            clf.fit(X[valid_idx], y[valid_idx])
            #yhat = clf.predict_proba(X[train_idx])
            #res4.append(log_loss(y[train_idx], yhat))
            res4.append(clf.score(X[train_idx], y[train_idx]))
        Rand.append(res4)
        pickle.dump(Rand, open("BTCRandAccuracyFull37.pkl", 'w'))
        j += 1

## 5. GP
params_dist = {
        'mi': UniformInt(5, 36),
        's' : Uniform(.4, 1.),
        'md': UniformInt(4, 11),
        'r' : Uniform(.4, 1.),
        'c' : Uniform(.4, 1.),
        }

if False:
    GP = []; j = 0
    kcv = StratifiedKFold(y, n_folds = 62, random_state = 1)
    logger.info('Starting Search...\n')
    for train_idx, valid_idx in kcv:
        def f(mi, s, md, r, c):
            clf = BoostedTreesClassifier(max_iterations = mi,
                    step_size = s,
                    max_depth = md,
                    row_subsample = r,
                    column_subsample = c,
                    verbose = 0)
            clf.fit(X[valid_idx], y[valid_idx])
            #yhat = clf.predict_proba(X[train_idx])
            #return -log_loss(y[train_idx], yhat)
            return clf.score(X[train_idx], y[train_idx])
        gp = GPUCBOpt(kernel = DoubleExponential, max_iter = 243,
                mu_prior = -1, sigma_prior = .20, sig = .005, 
                n_grid = 1000, time_budget = 3600*10, verbose = 1)
        gp.fit(func = f, params_dist = params_dist)
        GP.append(gp.y)
        pickle.dump(GP, open("BTC_GP_LLFull.pkl", 'w'))
        j += 1
