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

selected_model = os.environ['model_feat']
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

N_TREES = 1000
nCores = int(os.environ['OMP_NUM_THREADS'])

INITIAL_PARAMS = {
        'LogisticRegression':             {},
        'RandomForestClassifier':         {'n_estimators' : N_TREES},
        'ExtraTreesClassifier':           {'n_estimators' : N_TREES, 
                                           'n_jobs' : nCores},
        'SGDClassifier':                  {'penalty' : 'elasticnet'},
        'AdaBoostClassifier':             {},
        'SVC':                            {'probability' : True},
        'GradientBoostingClassifier':     {},
        'MultilayerPerceptronClassifier': {'activation' : 'relu'},
        'MultinomialNB':                  {}
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
        'MultinomialNB':                 {
            'alpha'         : [.1, .2, .5, 1.]
            }
        }

LogLoss = make_scorer(LogLossAdjGrid, greater_is_better = False, 
                      needs_proba = True)
Accuracy = make_scorer(accuracy_score, greater_is_better = True, 
                      needs_proba = False)

def FindParams(model, feature_set, y, subsample = None, grid_search = True):
    """
    Return parameter set for the model, either found through cross validation
    grid search, or load from file
    """
    # ipdb.set_trace()
    model_name = model.__class__.__name__
    if model.__class__.__name__ == 'SGDClassifier':
        scorer = Accuracy # SGD can not predict probability
    else:
        scorer = LogLoss
    if model.__class__.__name__ == 'ExtraTreesClassifier':
        nCores = 1
    params = INITIAL_PARAMS.get(model_name, {})
    model.set_params(**params)
    y = y if subsample is None else y[subsample]
    model_feat = stringify(model, feature_set)
    logger.info("Start RandomizedSearchCV paramaeter for %s",
                model_feat)
    try:
        with open('../Params/RandomizedSearchCV/%s_saved_params.json' 
                  % model_feat) as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}


    if (grid_search and stringify(model, feature_set) not in saved_params):
        X, _ = GetDataset(feature_set)
        clf = RandomizedSearchCV(model, PARAM_GRID[model_name], 
                scoring = scorer, cv = 5, n_iter = nGrids,
                n_jobs = nCores, random_state = 314, verbose = 2) 
        clf.fit(X, y)
        logger.info("Found params (%s > %.4f): %s" %(
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_))
        params.update(clf.best_params_)
        saved_params[stringify(model, feature_set)] = params
        with open('../Params/RandomizedSearchCV/%s_saved_params.json' 
                  % model_feat, 'w') as f:
            json.dump(saved_params, f, indent = 4, separators = (',', ': '),
                      ensure_ascii = True, sort_keys = True)
    else:
        params.update(saved_params.get(stringify(model, feature_set), {}))
        if grid_search:
            logger.info("Using params %s: %s" % (model_feat, params))

    return params

if __name__ == '__main__':
    SEED = 314
    nGrids = 50 
    #selected_model = "LR_text"
    #nCores = 8
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
                   'MNB'  : MultinomialNB
                 }
    model_id, dataset = selected_model.split('_')
    model = model_dict[model_id]()
    
    if model_id != "MNB": model.set_params(random_state = SEED)
     
    FindParams(model, dataset, y)
