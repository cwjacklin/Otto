import pandas as pd
import numpy as np
import logging
from utils import *
from sklearn.preprocessing import StandardScaler
logger = logging.getLogger(__name__)

def FormatData():
    np.random.seed(0)
    train = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')

    index = np.arange(len(train))
    np.random.shuffle(index)
    train = train.loc[index]
    train.index = np.arange(len(train))

    y = train.target.values
    X = train.drop(['id', 'target'], axis = 1).as_matrix()
    Xtest = test.drop('id', axis = 1).as_matrix()

    cut = len(y)*7/10
    train = np.arange(cut)
    valid = np.arange(cut, len(y))
    np.savez_compressed(file = "../Data/Data.npz", X = X, Xtest = Xtest, y = y,
                        train = train, valid = valid)

def LoadData():
    data = np.load("../Data/Data.npz")
    return data['X'], data['y'], data['Xtest']

def LoadFromCache(filename, use_cache = True):
    """Attempt to load data from cache."""
    data = (None, None, None)
    try:
        with open("../Data/%s.npz" % filename, 'r') as f:
            data = np.load(f)
            X, Xtest = data['X'], data['Xtest']
    except IOError:
            pass
    return X, Xtest

def SaveDataset(filename, X, Xtest):
    """Save the engineered features"""
    try:
        with open("../Data/%s.npz" % filename, 'r') as f:
            logging.warning("Dataset %s already existed", filename)
    except IOError:
            logger.info("> saving %s to disk", filename)
            np.savez_compressed("../Data/" + filename, X = X, Xtest = Xtest)

def GetDataset(feature_set = 'original', train = None, valid = None,
        ensemble_list = None):
    logger.info("Loading Feature Set %s", feature_set)
    if feature_set == 'ensemble':
        list_yhat = []
        for model_name in ensemble_list:
            file_name = "../Submission/yhat_" + model_name + "_full.npz"
            list_yhat.append(np.load(file_name)['yhat'])
        X = np.hstack(list_yhat)
        eps = 1e-6; X[X < eps] = eps; X[X > 1 - eps] = 1 - eps
        X = np.log(X/(1 - X))
        list_yhat = []
        for model_name in ensemble_list:
            file_name = "../Submission/yhat_" + model_name + "_test.npz"
            list_yhat.append(np.load(file_name)['yhat'])
        Xtest = np.hstack(list_yhat)
        Xtest[Xtest < eps] = eps; Xtest[Xtest > 1 - eps] = 1 - eps
        logger.info("Transforming log(p/(1-p))")
        Xtest = np.log(Xtest/(1 - Xtest))
        if train is not None:
            Xtest = X[valid]
            X = X[train]
    else:
        if len(feature_set.split("_")) > 1:
            logger.error("Please change featureset name from _ to -")
        feat = feature_set.split('-')
        try:
            with open("../Data/%s.npz" % feat[0], 'r') as f:
                data = np.load(f)
                if train is None:
                    X, Xtest = data['X'], data['Xtest']
                else:
                    X = data['X']
                    if valid is None:
                        valid = [i for i in xrange(X.shape[0]) if i 
                                    not in train]
                    Xtest = X[valid, :]
                    X = X[train, :]
        except IOError:
                logging.warning("Could not find feature set %s", feature_set)
                return False
        if len(feat) > 1 and feat[1] == 'standardized':
            logger.info("Standardizing Feature...")
            clf = StandardScaler()
            clf.fit(X)
            Xtest = clf.transform(Xtest)
            X = clf.transform(X)
        if feat[-1] == 'square':
            X = AddSquare(X)
            Xtest = AddSquare(Xtest)
            clf.fit(X)
            Xtest = clf.transform(Xtest)
            X = clf.transform(X)
    logger.info("X shape: (%d, %d). Xtest shape: (%d, %d)", np.shape(X)[0],
            np.shape(X)[1], np.shape(Xtest)[0], np.shape(Xtest)[1])
    return X, Xtest

def CreateDataset(X, Xtest, y, datasets = []):
    for dataset in datasets:
        if   dataset == 'text':
            X, Xtest = TextTransform(X, Xtest)
        elif dataset == 'log':
            X, Xtest = np.log10(X + 1), np.log10(Xtest + 1)
        elif dataset == 'original':
            pass
        else:
            logging.warning("Datasets must be one of: text, original, log")
        SaveDataset(dataset, X, Xtest)
