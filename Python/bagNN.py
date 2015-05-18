from ml import *

if False:
    X, Xtest = GetDataset("ensemble", ensemble_list =
        ['btc','btc2','btc3','btc4','svc','svc2','svc3','nn','nn2','nic',
            'mpc','knc','etc','cccv', 'log'])

    clf = MultilayerPerceptronClassifier()

    clf.set_params( learning_rate       = 'constant', 
                    hidden_layer_sizes  = 899, 
                    max_iter            = 993, 
                    power_t             = .9247,
                    alpha               = .005889, 
                    learning_rate_init  = 0.001009,
                    verbose             = 1)

if False:
    clf.set_params( learning_rate       = 'invscaling', 
                    hidden_layer_sizes  = 121, 
                    max_iter            = 929, 
                    power_t             = .8935,
                    alpha               = .004281, 
                    learning_rate_init  = 0.3668,
                    verbose             = 1)

    seed = 0
    np.random.seed(seed)
    list_yhat = []
    for i in xrange(20):
        print i
        train = np.random.choice(np.arange(len(X)), size = len(X)*4/5, replace = False)
        clf.fit(X[train], y[train])
        list_yhat.append(clf.predict_proba(Xtest))
    X = np.hstack(list_yhat)
    np.savez_compressed('./Lasagne/yhatMPC15.npz', yhat = X)

if False:
    X, Xtest = GetDataset("ensemble", ensemble_list =
        ['btc','btc2','btc3','btc4','svc','svc2','svc3','nn','nn2','nic',
            'mpc','knc','etc','log', 'keras', 'cccv', 'crfcbag', 'cetcbag'])
    clf = MultilayerPerceptronClassifier()
    clf.set_params( learning_rate       = 'invscaling', 
                    hidden_layer_sizes  = 361, 
                    max_iter            = 1451, 
                    power_t             = .7459,
                    alpha               = .006976, 
                    learning_rate_init  = 0.1722,
                    verbose             = 1)

    list_yhat = []
    for i in xrange(5):
        np.random.seed(i)
        print i
        train = np.random.choice(np.arange(len(X)), size = len(X)*4/5, replace = False)
        clf.fit(X[train], y[train])
        list_yhat.append(clf.predict_proba(Xtest))
    X = np.hstack(list_yhat)
    np.savez_compressed('./Lasagne/yhatMPC19.npz', yhat = X)

def GetYhat(model, feature_set, y):
    kcv = StratifiedKFold(y, 5, shuffle = True)
    X, Xtest = GetDataset(feature_set)
    res_cv = np.empty((len(y), len(np.unique(y))))
    res_test = 0
    for train_idx, valid_idx in kcv:
        model.fit(X[train_idx], y[train_idx])
        res_cv[valid_idx] = model.predict_proba(X[valid_idx])
        res_test = res_test + model.predict_proba(Xtest)
    return res_cv, res_test/5

print 'qwe'
if False:
    X, y, Xtest = LoadData()
    clf = CalibratedClassifierCV(
              base_estimator = ExtraTreesClassifier(
                  n_estimators = 300, 
                  max_features = 9, verbose = 0, n_jobs = -1), 
              method = 'isotonic',
              cv = 10)
    np.random.seed(1)
    res_cv, res_test = 0, 0
    n_bag = 100
    for i in xrange(n_bag):
        yhat_cv, yhat_test = GetYhat(clf, 'original', y)
        res_cv   = res_cv   + yhat_cv
        res_test = res_test + yhat_test
        print i, log_loss(y, yhat_cv)
    res_cv = res_cv/n_bag
    res_test = res_test/n_bag
    np.savez_compressed('../Submission/yhat_cetcbag_full.npz', yhat = res_cv)
    np.savez_compressed('../Submission/yhat_cetcbag_test.npz', yhat = res_test)
if False:
    X, y, Xtest = LoadData()
    from gl import BoostedTreesClassifier
    import json
    import time
    params = json.load(open('../Params/Best/BTC:original_saved_params.json'))
    params = params['BTC:original']
    np.random.seed(1)
    clf = BoostedTreesClassifier(**params)
    res_cv, res_test = 0, 0
    n_bag = 10

    time_before = time.time()
    for i in xrange(n_bag):
        yhat_cv, yhat_test = GetYhat(clf, 'original', y)
        res_cv   = res_cv   + yhat_cv
        res_test = res_test + yhat_test
        print i, log_loss(y, yhat_cv)
        if time.time() - time_before > 9*3600: break
    res_cv = res_cv/(i+1)
    res_test = res_test/(i+1)
    np.savez_compressed('../Submission/yhat_btcbag_full.npz', yhat = res_cv)
    np.savez_compressed('../Submission/yhat_btcbag_test.npz', yhat = res_test)
if True:
    pass
