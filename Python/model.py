import itertools
import numpy as np
import sys
import time
import pandas as pd
import pickle
import os

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
from logit import  *

def write(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def getGrid(center, length = 5, scale = 2):
    return center*(scale + 0.)**np.arange(-length, length)

def getSubmission(file_name, model, X, y, Xtest, **kwargs):
    time_before = time.time()
    md = model(**kwargs)
    write("Start Training " + str(model).split(" ")[-1].split(".")[-1][:-2] + \
          "\n" + str(kwargs) + "\n")
    md.fit(X, y)
    yhat = md.predict_proba(Xtest)
    yhat = (yhat + 0.001)/1.009
    submission = pd.read_csv('../Data/sampleSubmission.csv')
    yhat = pd.DataFrame(yhat, index = submission.id.values, 
                        columns = submission.columns[1:])
    yhat.to_csv(file_name, index_label = 'id')

def runModel(model, metric, X, y, train, valid, **kwargs):
    time_before = time.time()
    md = model(**kwargs)
    write("Start Training " + str(model).split(" ")[-1].split(".")[-1][:-2] + \
          "\n" + str(kwargs) + "\n")
    md.fit(X[train], y[train])
    write("Finish Training, Fit Valid Data\n")
    yhat = md.predict_proba(X[valid])
    #acc = accuracy_score(y[valid], yhat)
    acc = metric(y[valid], (yhat + 0.01)/1.09)
    write(str(metric).split(" ")[1] + "\t\t: " + str(acc) + "\n")
    running_time = time.time() - time_before
    write("Running time: " + str(running_time) + "\n\n\n")
    return acc


def LoadData():
    write("Loading Data...\n")
    data = np.load("../Data/Data.npz")
    global train; train = data['train']
    global valid; valid = data['valid']
    global X    ; X     = data['X']
    global y    ; y     = data['y']
    global Xtest; Xtest = data['Xtest']

def gridSearch(model, scorer, params, X, y, n_jobs):
    write("Grid Searching: " + str(model) +
          " with scorer: " + str(scorer) + "\n")
    write("\nParameters searched: " + str(params) + "\n")
    grid   = GridSearchCV(model, params, scorer, cv = 5, n_jobs = n_jobs, 
                          verbose = 2)
    grid.fit(X, y)
    write("Best Score: " + str(grid.best_score_)  + " by " + 
                           str(grid.best_params_) + " in " + 
                           str(grid.best_estimator_) + "\n")
    return grid    

def logLossAdj(y, yhat, inflate = 0.001):
    return log_loss(y, (yhat + inflate)/(1 + 9*inflate))

def logLossAdjGrid(y, yhat):
    return np.min([logLossAdj(y, yhat, eps = inflate) for inflate in 
           getGrid(0.0001, 10)])

def mainGrid(X, to_run = ["svm"]):
    ### Define Objective
    log_scorer = make_scorer(log_loss, greater_is_better = False, 
                         needs_proba = True) 
    acc_scorer = make_scorer(accuracy_score, greater_is_better = True,
                         needs_proba = False)
    log_sc_adj = make_scorer(logLossAdjGrid, greater_is_better = False, 
                         needs_proba = True)
    
    ### 0. Feature Engineer
    X = TextTransform(X)
    #X = AddSquare(X)

    ### 1. Logistic
    if "logit" in to_run:
        global logGrid 
        model   = LogisticRegression()
        params  = dict(penalty = ['l1', 'l2'], C = 2.**np.arange(-4,4))
        logGrid = gridSearch(model, log_scorer, params, X, y, 
                             n_jobs = nCores)
    
    ### 2. Random Forest
    if "rf" in to_run:
        global rfmGrid
        model   = RandomForestClassifier(n_estimators = 1000)
        params  = dict(max_features = 10*np.arange(1,9))
        rfmGrid = gridSearch(model, log_sc_adj, params, X, y, n_jobs = nCores)
        model   = ExtraTreesClassifier(n_estimators = 1000)
        rfmGrid = gridSearch(model, log_sc_adj, params, X, y, n_jobs = nCores)

    ### 3. Stochastic Gradient Descent
    if "sgd" in to_run:
        global sgdGrid
        model   = SGDClassifier(penalty = 'elasticnet', fit_intercept = True,
                              n_iter = 10, verbose = 0, n_jobs = 1, 
                              warm_start = True)
        params  = dict(loss = ['hinge', 'log', 'modified_huber', 'perceptron'], 
                      alpha = 1e-4*2.**np.arange(-8,8),
                   l1_ratio = [0, 0.25, 0.5, 0.75, 1])
        sgdGrid = gridSearch(model, acc_scorer, params, X, y, n_jobs = nCores)

    ### 4. Adaboost
    if "ada" in to_run:
        global adaGrid
        model   = AdaBoostClassifier()
        params  = dict(n_estimators = [50, 100, 200, 400],
                      learning_rate = 1e-1*2.**np.arange(-5,5))
        adaGrid = gridSearch(model, log_sc_adj, params, X, y, n_jobs = nCores)

    ### 5. Support Vector Machine
    if "svm" in to_run:
        global svmGrid
        model   = SVC(kernel = 'poly', probability = True)
        params  = dict(C = [8., 4., 2., 1., .5, .2, .1],
                  kernel = ['poly', 'rbf'],
                   gamma = [.1, .2, .5, 1., 2., 4., 8.])
        svmGrid = gridSearch(model, log_sc_adj, params, X, y, n_jobs = nCores)


    ### 6. Gradient Boosting Machine
    if "gbm" in to_run:
        global gbmGrid
        model   = GradientBoostingClassifier(n_estimators = 100, 
                         warm_start = True)
        params  = dict(learning_rate = 0.1*2.**np.arange(-8,8),
                          subsample = [0.6, 0.8, 1.0])
        gbmGrid = gridSearch(model, log_sc_adj, params, X, y, n_jobs = nCores)

    ### 7. Naive Bayes
    if "nb" in to_run:
        global nbGrid
        model   = GaussianNB()
        params  = dict(alpha = getGrid(0.001, 10))
        params  = dict()
        nbGrid  = gridSearch(model, log_sc_adj, params, X, y, n_jobs = nCores)

if __name__ == "__main__":
    try: X
    except NameError: LoadData()
    try: model = os.environ['model']
    except KeyError: model = ''
    nCores = int(os.environ['OMP_NUM_THREADS'])
    print nCores
    print model
    mainGrid(X, to_run = [])

    rbm = BernoulliRBM(n_components = 100, learning_rate = 0.1, 
                       batch_size = 128, n_iter = 20, 
                       verbose = True)
