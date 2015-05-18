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
import multiprocessing as mp
from utils                      import *
from data                       import LoadData, GetDataset
from scipy.stats                import randint, uniform
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
from nolearn.dbn                import DBN
from sklearn.calibration        import CalibratedClassifierCV
sys.path.insert(0, '../Library/MLP/')
sys.path.insert(0, '../Library/CMC/')
sys.path.insert(0, '../Library/GPUCB')
from multilayer_perceptron      import MultilayerPerceptronClassifier
from CMC                        import ConstrainedMultinomialClassifier
from CMC                        import GetBounds
from sklearn.linear_model       import LogisticRegressionCV
from gpucb                      import *
from randomDirection            import *
###############################################################################
### 1. Setting Things Up
###############################################################################

try:
    selected_model = os.environ['model_feat']
except KeyError:
    selected_model = "CCCV_original"
    Write("No model selected. Use default model: %s\n" % selected_model)

try:
    job_id = os.environ['job_id']
except KeyError:
    job_id = "72"
    Write("No jobid provided. Use default %s\n" % job_id)

try:
    nCores = int(os.environ['OMP_NUM_THREADS'])
except ValueError:
    nCores = 12

SEED = int(job_id)
N_TREES = 1000

CONFIG = {}
CONFIG['nCores'] = nCores
CONFIG['SEED']   = int(job_id)

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
        'BoostedTreesClassifier'        : {'verbose' : False},
        'KNeighborsClassifier'          : {
            'weights'                   : 'uniform',
            'leaf_size'                 : 1000
            },
        'ConstrainedMultinomialClassifier': {},
        'DBN'                           : {'verbose' : 2},
        'CalibratedClassifierCV'        : {
            'base_estimator'            : ExtraTreesClassifier(n_jobs = -1),
            'cv'                        : 10,
            'method'                    : 'isotonic',
            }
        }

PARAM_GRID = {
        'LogisticRegression':             { 
            'C'             : np.logspace(-20, 10, num = 210, base = 2.), 
            'penalty'       : ['l1', 'l2'],
            'class_weight'  : [None, 'auto']
            },
        'RandomForestClassifier':         {
            'max_features'  : GetGrid(  50,  4, mode = "add", scale = 10),
            'criterion'     : ['gini', 'entropy'], 
            'max_depth'     : [None, 6, 8, 10]
            },
        'ExtraTreesClassifier':           {
            'max_features'  : randint(5,94),
            'criterion'     : ['gini', 'entropy'],
            'max_depth'     : randint(4,20),
            'min_samples_split'                 : randint(1, 21),
            'min_samples_leaf'                  : randint(1, 21),
            'n_estimators'  : randint(100,1000)
            },
        'CalibratedClassifierCV':         {
            #'base_estimator__criterion'         : ['gini', 'entropy'],
            #'base_estimator__bootstrap'         : [True, False],
            #'base_estimator__max_depth'         : randint(4,20),
            #'base_estimator__min_samples_split' : randint(1, 21),
            #'base_estimator__min_samples_leaf'  : randint(1, 21),
            'cv'                                : randint(3,15),
            'base_estimator__max_features'      : randint(10,93),
            'base_estimator__n_estimators'      : randint(100, 1000)
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
            'C'             : np.logspace(-10,10, base = 2),
            'gamma'         : np.logspace(-4,4, base = 2),
            'kernel'        : ['rbf']
            },
        'GradientBoostingClassifier':     {
            'learning_rate' : [.1, .2, .5, 1.],
            'n_estimators'  : [50, 100, 200, 400],
            'max_depth'     : [4, 8, 12], 
            'max_features'  : GetGrid(  50,  4, mode = "add", scale = 10)
            },
        'MultilayerPerceptronClassifier': {
            'max_iter'      : randint(100,1000),
            'hidden_layer_sizes' : randint(100,1000),
            'alpha'         : np.logspace(-15,0,100, base = 2),
            'learning_rate' : ['constant', 'invscaling'],
            'learning_rate_init': np.logspace(-10,0,base = 2),
            'power_t'       : uniform(.3,.69),
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
            },
        'DBN'                             :{
            'layer_sizes'   : np.vstack([-np.ones(9, dtype = np.int16), 
                                np.logspace(6, 10, 9, base = 2),
                                         -np.ones(9, dtype = np.int16)]).T,
            'scales'        : np.logspace(-10,1, 12, base = 2),
            'learn_rates'   : np.logspace(-8, 0, 17, base = 2),
            'use_re_lu'     : [True, False],
            'learn_rate_decays': [1., .99, .95, .90],
            'learn_rate_minimums': np.logspace(-20,-10, base = 2),
            'l2_costs'      : np.logspace(-20,-6, 15, base = 2),
            'epochs'        : np.logspace(6, 11, 11, base = 2)
            }
        }

###############################################################################
### 2. Function to Randomized Grid Search CV for best parameters
###############################################################################

logLoss = make_scorer(LogLossAdjGrid, greater_is_better = False, 
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
            'KNeighborsClassifier', 'AdaBoostClassifier']:
        scorer = Accuracy # SGD can not predict probability
    else:
        scorer = logLoss
    if model.__class__.__name__ in ['ExtraTreesClassifier', 
        'BoostedTreesClassifier', 'MultilayerPerceptronClassifier', 'DBN',
        'CalibratedClassifierCV']:
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
                (nCores, CONFIG['nGrids'], job_id))
    logger.info("Scorer: %s", scorer.__class__.__name__)
    try:
        with open('../Params/RandomizedSearchCV/%s_saved_params.json' 
                  % model_feat) as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}

    if (grid_search and stringify(model, feature_set) not in saved_params):
        ### Fit Model
        X, _ = GetDataset(feature_set, 
                        ensemble_list = CONFIG['ensemble_list'])
        clf = RandomizedSearchCV(model, PARAM_GRID[model_name], 
                scoring = scorer, cv = 5, n_iter = CONFIG['nGrids'],
                n_jobs = nCores, random_state = CONFIG['SEED'], verbose = 2) 
        clf.fit(X, y)
        
        ### Reporting
        logger.info("Found params (%s > %.4f): %s" %(
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_))
        #ipdb.set_trace()
        for fit_model in clf.grid_scores_:
            logger.info("MeanCV: %.4f", fit_model[1])
            for para, para_value in fit_model[0].iteritems():
                if para != 'bounds':
                    logger.info("%20s: %10s", para, para_value)
                else:
                    logger.info("Bound with length %d: ", len(para_value))
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
    pickle.dump(clf.grid_scores_, open('../Employment/MPC15/' + \
                   model_feat + job_id + '.pkl', 'w'))
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

    X, Xtest = GetDataset(feature_set, train, valid, 
            ensemble_list = CONFIG['ensemble_list'])
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

def ReportPerfCV(model, feature_set, y, calibrated = False, n_folds = 5, 
                    short = False):
    kcv = StratifiedKFold(y, n_folds, shuffle = True); i = 1
    res = np.empty((len(y), len(np.unique(y))))
    X, Xtest = GetDataset(feature_set)
    if calibrated: 
        logger.info("Enabling probability calibration...")
        model = CalibratedClassifierCV(model, 'sigmoid', cv = n_folds - 1)
    for train_idx, valid_idx in kcv:
        logger.info("Running fold %d...", i);
        model.fit(X[train_idx], y[train_idx])
        logger.info("Fold %i Accuracy: %.4f", i, 
                model.score(X[valid_idx], y[valid_idx]))
        res[valid_idx, :] = model.predict_proba(X[valid_idx])
        logger.info("Fold %i Log Loss: %.4f", i, 
                log_loss(y[valid_idx], res[valid_idx]))
        i += 1
        if short: break
    if short: return -log_loss(y[valid_idx], res[valid_idx])
    yhat = np.argmax(res, axis = 1) + 1
    Y    = np.array([int(i[-1]) for i in y])
    logger.info("CV Accuracy: %.5f", accuracy_score(Y, yhat))
    logger.info("CV Log Loss: %.4f", log_loss(y, res))
    return res, -log_loss(y, res)

_, y, _ = LoadData(); del _
CONFIG['ensemble_list'] = ['btc', 'btc2', 'svc', 'mpc', 'etc', 'knc', 'nn']


def OptSVC(C, gamma):
    model = SVC(C = C, gamma = gamma, probability = True)
    return ReportPerfCV(model, "text", y)

def OptBTC(step_size = .5, max_iterations = 100, row_subsample = .9, 
        column_subsample = .9,
        max_depth = 8):
    model = BoostedTreesClassifier(step_size = step_size, 
            max_iterations = max_iterations,
            row_subsample = row_subsample, column_subsample = column_subsample,
            max_depth = max_depth)
    logger.info("Params: %s", model.get_params())
    return ReportPerfCV(model, "original", y) 

if __name__ == '_main__':
    logger.info("Running %s, on %d cores" %(selected_model, nCores))
    _, y, _ = LoadData(); del _
    CONFIG['ensemble_list'] = ['btc','btc2','btc3','btc4','svc','svc2','svc3',
                             'nn','nn2', 'nic', 'mpc','knc','etc','log',
                             'keras', 'cccv', 'crfcbag', 'cetcbag'] 
    model_dict = { 'LR'   : LogisticRegression,
                   'RFC'  : RandomForestClassifier,
                   'ETC'  : ExtraTreesClassifier,
                   'ABC'  : AdaBoostClassifier,
                   'SVC'  : SVC,
                   'SGDC' : SGDClassifier,
                   'GBC'  : GradientBoostingClassifier,
                   'MPC'  : MultilayerPerceptronClassifier,
                   'MNB'  : MultinomialNB,
                   'KNC'  : KNeighborsClassifier,
                   'CMC'  : ConstrainedMultinomialClassifier,
                   'DBN'  : DBN,
                   'CCCV' : CalibratedClassifierCV,
                 }
    if selected_model[:3] == "BTC": 
        from gl import BoostedTreesClassifier
        model_dict['BTC'] = BoostedTreesClassifier
        PARAM_GRID['BoostedTreesClassifier'] = {
                   'max_iterations': randint(300,3000),
                        'step_size': np.logspace(-4,  0,500),
                    'row_subsample': uniform(.4, .6),
                 'column_subsample': uniform(.1, .9),
                        'max_depth': randint(5,40)}
   
    model_id, dataset = selected_model.split('_')
    model = model_dict[model_id]()

    if model_id in ['LR','CMC']:
        CONFIG['nGrids'] = 500
    elif model_id in ['RFC', 'ETC', 'GBC', 'MPC', 'BTC', 'CCCV']:
        CONFIG['nGrids'] = 30
    else:
        CONFIG['nGrids'] = 30
    if 'random_state' in model.get_params(): 
        model.set_params(random_state = 1)
    logger.debug('\n' + '='*50)
    res = FindParams(model, dataset, y, CONFIG)

if __name__ == '_main__':
    #res = GPUCB(func = OptSVC, kernel = Matern32, 
    #        params_dist = {'C': Uniform(0,20), 'gamma': Uniform(1, 20)}, 
    #        sig = .005, mu_prior = -.5, sigma_prior = .05, 
    #        n_iter = int(job_id), n_grid = 1000, seed = int(job_id),
    #        time_budget = int(job_id)*3600)
    res = GPUCB(func = OptBTC, kernel = Matern32,
            params_dist = {'max_iterations': UniformInt(500,2500),
                            'step_size': LogUniform(.001,.1),
                            'row_subsample': Uniform(.4,1.),
                            'column_subsample': Uniform(.1, 1.),
                            'max_depth': UniformInt(5,40)
                            },
            sig = .005, mu_prior = -.5, sigma_prior = .05,
            n_iter = 1000, n_grid = 1000, seed = int(job_id),
            time_budget = int(job_id)*3600)
    print res
    pickle.dump(res, file = open(selected_model + job_id + ".pkl", 'w'))

if __name__ == '_main__':
    clf = GPUCBOpt(kernel = Matern32W(invrho = 20), max_iter = 1000,
                mu_prior = -.63, sigma_prior = .10, sig = .005, 
                n_grid = 1000, random_state = int(job_id), 
                time_budget = 3600*int(job_id), verbose = 1, file_id = job_id)
    params_dist = {'max_iterations': UniformInt(300,3000),
                        'step_size': LogUniform(.0001,.1),
                    'row_subsample': Uniform(.4,1.),
                 'column_subsample': Uniform(.1, 1.),
                        'max_depth': UniformInt(5,40)}

    clf.fit(func = OptBTC, params_dist = params_dist)

if __name__ == '__main__':
    param_distributions =        {
            'max_iterations'    : UniformInt(100,3000),
            'step_size'         : LogUniform(.0001, 1.),
            'max_depth'         : UniformInt(2,50),
            'row_subsample'     : Uniform(.3,1.),
            'column_subsample'  : Uniform(.3, 1.),
            'min_child_weight'  : LogUniform(.01,100),
            'min_loss_reduction': Uniform(0.0001,10)
            }
    from gl import BoostedTreesClassifier
    logger2 = logging.getLogger('graphlab')
    logger2.setLevel(logging.CRITICAL)
    CONFIG['ensemble_list'] = ['btc','btc2','btc3','btc4','svc','svc2','svc3',
            'nn','nn2','nic', 'mpc','knc','etc','cccv', 'log','crfcbag',
            'cetcbag', 'keras']
    X, Xtest = GetDataset('ensemble', ensemble_list = CONFIG['ensemble_list'])
    print "lkjr"
    clf = GaussianProcessCV(
            estimator           = BoostedTreesClassifier(verbose = False),
            param_distributions = param_distributions,
            kernel              = DoubleExponential,
            scoring             = LogLoss, 
            mu_prior            = -1.,
            sigma_prior         = .30,
            sig                 = .01,
            cv                  = 5, 
            max_iter            = 55,
            random_state        = 1, 
            time_budget         = 24*3600)
    clf.fit(X, y)
    #clf = RandomSearchCV(estimator = BoostedTreesClassifier(verbose = False),
    #        param_distributions = param_distributions,
    #        cv = 5, max_iter = 100, random_state = int(job_id))
    #clf.fit(X[:5000], y[:5000])

if __name__ == '_main__':
    param_distributions = {
            'max_iter'              : UniformInt(100,1500),
            'hidden_layer_sizes'    : UniformInt(100,1000),
            'alpha'                 : LogUniform(.001,0.01),
            'learning_rate_init'    : Uniform(.1,1.),
            'power_t'               : Uniform(.5,.99),
            }
    CONFIG['ensemble_list'] = ['btc','btc2','btc3','btc4','svc','svc2','svc3',
                             'nn','nn2', 'nic', 'mpc','knc','etc','log',
                             'keras', 'cccv', 'crfcbag', 'cetcbag'] 
    X, Xtest = GetDataset('ensemble', ensemble_list = CONFIG['ensemble_list'])
    clf = GaussianProcessCV(
            estimator           = MultilayerPerceptronClassifier(
                            verbose         = False, 
                            learning_rate   = 'invscaling'),
            param_distributions = param_distributions,
            kernel              = DoubleExponential,
            scoring             = LogLoss, 
            mu_prior            = -.50,
            sigma_prior         = .10,
            sig                 = .001,
            cv                  = 5, 
            max_iter            = 100,
            random_state        = int(job_id), 
            time_budget         = 24*3600
            )
    clf.fit(X, y)
