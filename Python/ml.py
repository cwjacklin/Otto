"""
Otto Challenge
--------------

This code do the randomized grid search to find best parameters. Multiple 
models are supported. 

Author: Hoang Duong <hduong@berkeley.edu>. Many lines of code are copied
        from Paul Duan Amazon Employee Access Kaggle Winner Code.
"""

###############################################################################
### 0. Importing 
###############################################################################

import sys
import logging
import itertools
import cPickle as pickle
import json
import ipdb
import os

from utils                      import *
from data                       import LoadData, GetDataset

from sklearn.neural_network     import BernoulliRBM
from sklearn.ensemble           import RandomForestClassifier
from sklearn.svm                import SVC
from sklearn.qda                import QDA
from sklearn.lda                import LDA
from sklearn.ensemble           import AdaBoostClassifier
from sklearn.ensemble           import GradientBoostingClassifier
from sklearn.linear_model       import SGDClassifier, LogisticRegression
from sklearn.ensemble           import ExtraTreesClassifier
from sklearn.naive_bayes        import MultinomialNB
from sklearn.naive_bayes        import GaussianNB
from sklearn.metrics            import accuracy_score, log_loss, make_scorer
from sklearn.grid_search        import GridSearchCV
from sklearn.grid_search        import RandomizedSearchCV
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.cross_validation   import StratifiedKFold

sys.path.insert(0, '../Library/MLP/')
from multilayer_perceptron  import MultilayerPerceptronClassifier
#from gl                     import BoostedTreesClassifier

###############################################################################
### 1. Setting Things Up
###############################################################################

# selected_model = os.environ['model_feat']
try:
    selected_model = os.environ['model_feat']
except KeyError:
    selected_model = "MPC_log"
    Write("No model selected. Use default Logistic model\n")

try:
    job_id = os.environ['job_id']
except KeyError:
    job_id = "003"
    Write("No jobid provided. Use default %s\n" % job_id)
nCores = int(os.environ['OMP_NUM_THREADS'])
nCores = 12
nGrids = 10 


SEED = int(job_id)
N_TREES = 1000

CONFIG = {}
CONFIG['nCores'] = nCores
CONFIG['SEED']   = SEED
CONFIG['nGrids'] = nGrids

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
        filename="history.log", 
        filemode='a', level=logging.DEBUG,
        datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
        datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)


###############################################################################
### 2. Setting Initial Parameters
###############################################################################

INITIAL_PARAMS = {
        'LogisticRegression'            : {},
        'RandomForestClassifier'        : {'n_estimators' : N_TREES},
        'ExtraTreesClassifier'          : {'n_estimators' : N_TREES, 
                                           'n_jobs' : nCores},
        'SGDClassifier'                 : {'penalty' : 'elasticnet'},
        'AdaBoostClassifier'            : {},
        'SVC'                           : {'probability' : True},
        'GradientBoostingClassifier'    : {},
        'MultilayerPerceptronClassifier': {'activation' : 'relu'},
        'MultinomialNB'                 : {},
        'BoostedTreesClassifier'        : {'verbose' : True},
        'KNeighborsClassifier'          : {
            'weights'       : 'uniform',
            'leaf_size'     : 1000,
            'metric'        : 'minkowski'
            }

        }

PARAM_GRID = {
        'LogisticRegression':             { 
            'C'             : np.logspace(-4, 4, num = 9, base = 2.), 
            'penalty'       : ['l1', 'l2'],
            'class_weight'  : ['auto']
            },
        'RandomForestClassifier':         {
            'max_features'  : GetGrid(  50,  4, mode = "add", scale = 10),
            'criterion'     : ['gini', 'entropy'], 
            'max_depth'     : [None, 6, 8, 10]
            },
        'ExtraTreesClassifier':           {
            'max_features'  : GetGrid(  50,  4, mode = "add", scale = 10),
            'criterion'     : ['gini', 'entropy'],
            'max_depth'     : [None, 6, 8, 10]
            },
        'SGDClassifier':                  {
            'loss'          : ['hinge', 'log', 'modified_huber', 'perceptron'],
            'alpha'         : GetGrid(1e-3, 10, mode = "mul", scale = 2 ),
            'l1_ratio'      : [0, .25, .5, .75, 1.],
            'n_iter'      : [20, 40, 80, 160]
            },
        'AdaBoostClassifier':             {
            'learning_rate' : [.1, .2, .5, 1.],
            'n_estimators'  : [50, 100, 200, 400]
            },
        'SVC':                            {
            'C'             : np.arange(1,15),
            'gamma'         : np.arange(1,15),
            'kernel'        : ['rbf', 'poly', 'sigmoid']
            },
        'GradientBoostingClassifier':     {
            'learning_rate' : [.1, .2, .5, 1.],
            'n_estimators'  : [50, 100, 200, 400],
            'max_depth'     : [4, 8, 12], 
            'max_features'  : GetGrid(  50,  4, mode = "add", scale = 10)
            },
        'MultilayerPerceptronClassifier': {
            'max_iter'      : np.arange(20, 500),
            'hidden_layer_sizes' : np.arange(20, 1000),
            'alpha'         : np.logspace(-20,3,24, base = 2),
            'learning_rate' : ['constant', 'invscaling'],
            'learning_rate_init': [.1, .2, .5, 1.]
            },
        'MultinomialNB':                  {
            'alpha'         : [.1, .2, .5, 1.]
            },
        'BoostedTreesClassifier':         {
            'max_iterations': np.arange(150,501),
            'step_size'     : np.logspace(-5, 0, 6, base = 2),
            'max_depth'     : np.arange(6,25),
            'row_subsample' : [.5, .6, .7, .8, .9, 1.],
          'column_subsample': [.5, .6, .7, .8, .9, 1.],
          'min_child_weight': np.logspace(-10,5,16,base = 2),
        'min_loss_reduction': np.arange(20)
            },
        'KNeighborsClassifier':            {
            'n_neighbors'   : np.arange(5,500),
            'p'             : [1, 2],
            'metric'        : ['minkowski', 'canberra','hamming',
                                'braycurtis']
            }
        }

###############################################################################
### 2. Function to Randomized Grid Search CV for best parameters
###############################################################################

LogLoss = make_scorer(LogLossAdjGrid, greater_is_better = False, 
                      needs_proba = True)
Accuracy = make_scorer(accuracy_score, greater_is_better = True, 
                      needs_proba = False)

def FindParams(model, feature_set, y, CONFIG, subsample = None, 
                grid_search = True):
    """
    Return parameter set for the model, either found through cross validation
    grid search, or load from file
    """
    ### Setting configurations
    model_name = model.__class__.__name__
    if model.__class__.__name__ in ['SGDClassifier', 
            'KNeighborsClassifier']:
        scorer = Accuracy # SGD can not predict probability
    else:
        scorer = LogLoss
    if model.__class__.__name__ in ['ExtraTreesClassifier', 
        'BoostedTreesClassifier', 'MultilayerPerceptronClassifier']:
        nCores = 1
    else:
        nCores = CONFIG['nCores']
    ### Setting parameters
    params = INITIAL_PARAMS.get(model_name, {})
    model.set_params(**params)
    y = y if subsample is None else y[subsample]
    model_feat = stringify(model, feature_set)
    logger.info("Start RandomizedSearchCV paramaeter for %s",
                model_feat)
    logger.info("nCores: %d, nGrid: %d, job_id: %s" % 
                (nCores, nGrids, job_id))
    logger.info("Scorer: %s", scorer.__class__.__name__)
    try:
        with open('../Params/RandomizedSearchCV/%s_saved_params.json' 
                  % model_feat) as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}


    if (grid_search and stringify(model, feature_set) not in saved_params):
        ### Fit Model
        X, _ = GetDataset(feature_set)
        clf = RandomizedSearchCV(model, PARAM_GRID[model_name], 
                scoring = scorer, cv = 5, n_iter = CONFIG['nGrids'],
                n_jobs = nCores, random_state = CONFIG['SEED'], verbose = 3) 
        clf.fit(X, y)
        
        ### Reporting
        logger.info("Found params (%s > %.4f): %s" %(
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_))
        #ipdb.set_trace()
        for fit_model in clf.grid_scores_:
            logger.info("MeanCV: %.4f", fit_model[1])
            for para, para_value in fit_model[0].iteritems():
                logger.info("%20s: %10s", para, para_value)
        ### Save Parameters
        params.update(clf.best_params_)
        saved_params[stringify(model, feature_set)] = params
        with open('../Params/RandomizedSearchCV/%s_%s_saved_params.json' 
                  % (model_feat, job_id), 'w') as f:
            json.dump(saved_params, f, indent = 4, separators = (',', ': '),
                      ensure_ascii = True, sort_keys = True)
    else:
        params.update(saved_params.get(stringify(model, feature_set), {}))
        if grid_search:
            logger.info("Using params %s: %s" % (model_feat, params))

    return params

def GetPrediction(model, feature_set, y, train = None, valid = None, 
                    preds = "proba", verbose = 1):
    model_name = model.__class__.__name__
    params = INITIAL_PARAMS.get(model_name, {})
    model.set_params(**params)
    y = y if train is None else y[train]
    model_feat = stringify(model, feature_set)
    try:
        with open('../Params/Best/%s_saved_params.json' % model_feat) as f:
            saved_params = json.load(f).get(model_feat, {})
    except IOError:
        logging.warning("Could not find best parameter for %s with feature \
                set %s", model_name, feature_set)
        saved_params = {}
        return False
    
    for key in saved_params.keys():
        logger.info("%s: %s", key, saved_params[key])
        ### Fixing Unicode String issues
        if type(saved_params[key]) is unicode:
            saved_params[key] = str(saved_params[key])

    if 'verbose' in model.get_params():
        model.set_params(verbose = verbose)

    X, Xtest = GetDataset(feature_set, train, valid)
    model.set_params(**saved_params)
    logger.info("Fitting %s on %s feature", model_name, feature_set)
    model.fit(X, y)
    logger.info("Returning prediction")
    if preds == "proba":
        yhat = model.predict_proba(Xtest)
    elif preds == "class":
        yhat = model.predict(Xtest)
    else:
        logger.warning("preds must be either proba or class")
        return False
    return yhat

def GetPredictionCV(model, feature_set, y, CONFIG, n_folds = 5):
    kcv = StratifiedKFold(y, n_folds, random_state = CONFIG['SEED'])
    res = np.empty((len(y), len(np.unique(y)))); i = 1
    for train_idx, valid_idx in kcv:
        logger.info("Running fold %d...", i); i += 1
        res[valid_idx,:] = GetPrediction(model, feature_set, y, 
                                        train = train_idx, valid = valid_idx)
    return res

def GetLevel1(y, CONFIG):
    yhat_btc_full = np.load("yhat_btc_full.npz")['yhat']
    logger.info("BTC Log loss: %.4f" % log_loss(y, yhat_btc_full))
    yhat_btc_full2 = np.load("yhat_btc_full2.npz")['yhat']
    logger.info("BTC2 Log loss: %.4f" % log_loss(y, yhat_btc_full2))
    yhat_svc_full = np.load("yhat_svc_full.npz")['yhat']
    logger.info("SVC Log Loss: %.4f" % log_loss(y, yhat_svc_full))
    yhat_mpc_full = np.load("yhat_mpc_full.npz")['yhat']
    logger.info("MPC Log Loss: %.4f" % log_loss(y, yhat_mpc_full))
    X1 = np.hstack([yhat_btc_full, yhat_btc_full2, 
                    yhat_svc_full, yhat_mpc_full])
    X1 = np.log(X1/(1-X1))
    clf = LogisticRegression(C = .1, fit_intercept = False, 
            penalty = 'l1', random_state = CONFIG['SEED'])
    clf.fit(X1, y)
    
    yhat_btc_test = np.load("yhat_btc_test.npz")['yhat']
    yhat_btc_test2 = np.load("yhat_btc_test2.npz")['yhat']
    yhat_svc_test = np.load("yhat_svc_test.npz")['yhat']
    yhat_mpc_test = np.load("yhat_mpc_test.npz")['yhat']
    X1test = np.hstack([yhat_btc_test, yhat_btc_test2,
                        yhat_svc_test, yhat_mpc_test])
    X1test = np.log(X1test/(1 - X1test))
    yhat = clf.predict_proba(X1test)
    return yhat

    
if __name__ == '__main__':
    logger.info("Running %s, on %d cores" %(selected_model, nCores))
    _, y, _ = LoadData(); del _
    model_dict = { 'LR'   : LogisticRegression,
                   'RFC'  : RandomForestClassifier,
                   'ETC'  : ExtraTreesClassifier,
                   'ABC'  : AdaBoostClassifier,
                   'SVC'  : SVC,
                   'SGDC' : SGDClassifier,
                   'GBC'  : GradientBoostingClassifier,
                   'MPC'  : MultilayerPerceptronClassifier,
                   'MNB'  : MultinomialNB,
                   #'BTC'  : BoostedTreesClassifier,
                   'KNC'  : KNeighborsClassifier
                 }
    if selected_model[:3] == "BTC": 
        from gl import BoostedTreesClassifier
        model_dict['BTC'] = BoostedTreesClassifier

    model_id, dataset = selected_model.split('_')
    model = model_dict[model_id]()
    if model_id not in ["MNB", "BTC", "KNC"]: 
        model.set_params(random_state = SEED)
    logger.debug('\n' + '='*50)
    res = FindParams(model, dataset, y, CONFIG)
    """
    import multiprocessing as mp

    kcv = StratifiedKFold(y, 5, random_state = CONFIG['SEED'])
    idx = []
    for train_idx, valid_idx in kcv:
        idx.append((train_idx, valid_idx))

    def g(t):
        return GetPrediction(SVC(), "text", y, train  = t[0], valid = t[1])

    #pool = mp.Pool(processes = 5)
    #results = pool.map(g, idx)
    """
