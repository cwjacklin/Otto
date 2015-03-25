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

from utils                  import *
from data                   import *

from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble       import RandomForestClassifier
from sklearn.svm            import SVC
from sklearn.qda            import QDA
from sklearn.lda            import LDA
from sklearn.ensemble       import AdaBoostClassifier
from sklearn.ensemble       import GradientBoostingClassifier
from sklearn.linear_model   import SGDClassifier, LogisticRegression
from sklearn.ensemble       import ExtraTreesClassifier
from sklearn.naive_bayes    import MultinomialNB
from sklearn.naive_bayes    import GaussianNB
from sklearn.metrics        import accuracy_score, log_loss, make_scorer
from sklearn.grid_search    import GridSearchCV
from sklearn.grid_search    import RandomizedSearchCV

sys.path.insert(0, '../Library/MLP/')
from autoencoder            import *
from multilayer_perceptron  import *
from gl                     import BoostedTreesClassifier

###############################################################################
### 1. Setting Things Up
###############################################################################

# selected_model = os.environ['model_feat']
try:
    selected_model = os.environ['model_feat']
except KeyError:
    selected_model = "LG_text"
    Write("No model selected. Run Grid Search on default model\n")

try:
    job_id = os.environ['job_id']
except KeyError:
    job_id = "000"
    Write("No jobid provided. Run Grid Search with default job_id")
nCores = int(os.environ['OMP_NUM_THREADS'])
nGrids = 30 


SEED = int(job_id)
N_TREES = 1000

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
        filename="../Params/RandomizedSearchCV/%s.log" %selected_model, 
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
        'BoostedTreesClassifier'        : {'verbose' : True}
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
            'max_iter'      : [100, 200, 300],
            'hidden_layer_sizes' : [40, 80, 160, 320, 640],
            'alpha'         : GetGrid(1e-3, 10, mode = "mul", scale = 2 ),
            },
        'MultinomialNB':                  {
            'alpha'         : [.1, .2, .5, 1.]
            },
        'BoostedTreesClassifier':         {
            'max_iterations': np.arange(100,401),
            'step_size'     : np.logspace(-5, 0, 6, base = 2),
            'max_depth'     : np.arange(2,15),
            'row_subsample' : [.5, .6, .7, .8, .9, 1.],
          'column_subsample': [.5, .6, .7, .8, .9, 1.],
          'min_child_weight': np.logspace(-10,5,16,base = 2),
        'min_loss_reduction': np.arange(20)
            }
        }

###############################################################################
### 2. Function to Randomized Grid Search CV for best parameters
###############################################################################

LogLoss = make_scorer(LogLossAdjGrid, greater_is_better = False, 
                      needs_proba = True)
Accuracy = make_scorer(accuracy_score, greater_is_better = True, 
                      needs_proba = False)

def FindParams(model, feature_set, y, subsample = None, 
                grid_search = True):
    """
    Return parameter set for the model, either found through cross validation
    grid search, or load from file
    """
    model_name = model.__class__.__name__
    if model.__class__.__name__ == 'SGDClassifier':
        scorer = Accuracy # SGD can not predict probability
    else:
        scorer = LogLoss
    if model.__class__.__name__ in ['ExtraTreesClassifier', 
                                  'BoostedTreesClassifier']:
        nCores = 1
    params = INITIAL_PARAMS.get(model_name, {})
    model.set_params(**params)
    y = y if subsample is None else y[subsample]
    model_feat = stringify(model, feature_set)
    logger.info("Start RandomizedSearchCV paramaeter for %s",
                model_feat)
    logger.info("nCores: %d, nGrid: %d, job_id: %s" % 
                (nCores, nGrids, job_id))
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
                scoring = scorer, cv = 5, n_iter = nGrids,
                n_jobs = nCores, random_state = SEED, verbose = 2) 
        clf.fit(X, y)
        
        ### Reporting
        logger.info("Found params (%s > %.4f): %s" %(
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_))
        for fit_model in clf.grid_scores_:
            Write(str(fit_model))
        
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

if __name__ == '__main__':
    logger.info("Running Model %s, on %d cores" %(selected_model, nCores))
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
                   'BTC'  : BoostedTreesClassifier
                 }
    model_id, dataset = selected_model.split('_')
    model = model_dict[model_id]()
    if model_id not in ["MNB", "BTC"]: model.set_params(random_state = SEED)
    FindParams(model, dataset, y)
