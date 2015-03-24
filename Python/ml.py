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

logger = logging.getLogger(__name__)

N_TREES = 1000

INITIAL_PARAMS = {
        'LogisticRegression':             {},
        'RandomForestClassifier':         {'n_estimators' : N_TREES},
        'ExtraTreesClassifier':           {'n_estimators' : N_TREES},
        'SGDClassifier':                  {'penalty' : 'elasticnet'},
        'AdaBoostClassifier':             {},
        'SVC':                            {'probability' : True},
        'GradientBoostingClassifier':     {},
        'MultilayerPerceptronClassifier': {'activation' : 'relu'},
        'MultinomialNB':                  {}
        }

PARAM_GRID = {
        'LogisticRegression':             { 
            'C'             : GetGrid(   1,  4, mode = "mul", scale = 2 ), 
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
            'C'             : GetGrid(  1,   4, mode = "mul", scale = 2 ),
            'gamma'         : GetGrid(  1,   4, mode = "mul", scale = 2 ),
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
    params = INITIAL_PARAMS.get(model_name, {})
    model.set_params(**params)
    y = y if subsample is None else y[subsample]
    model_feat = stringify(model, feature_set)
    try:
        with open('../Params/%s_saved_params.json' % model_feat) as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}


    if (grid_search and stringify(model, feature_set) not in saved_params):
        X, _ = GetDataset(feature_set)
        clf = GridSearchCV(model, PARAM_GRID[model_name], 
                scoring = scorer, cv = 5, n_jobs = nCores, verbose = 2) 
        clf.fit(X, y)
        Write("Found params (%s > %.4f): %s\n" %(
                    stringify(model, feature_set),
                    clf.best_score_, clf.best_params_))
        params.update(clf.best_params_)
        saved_params[stringify(model, feature_set)] = params
        with open('../Params/%s_saved_params.json' % model_feat, 'w') as f:
            json.dump(saved_params, f, indent = 4, separators = (',', ': '),
                      ensure_ascii = True, sort_keys = True)
    else:
        params.update(saved_params.get(stringify(model, feature_set), {}))
        if grid_search:
            Write("Using params %s: %s" % (model_feat, params))

    return params

if __name__ == '__main__':
    SEED = 314
    selected_model = os.environ['model_feat']
    nCores = int(os.environ['OMP_NUM_THREADS'])
    Write("Running Model %s, on %d cores\n" %(selected_model, nCores))
    
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
    
