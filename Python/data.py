import pandas as pd
import numpy as np
import logging
from utils import *

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

def GetDataset(feature_set = 'original'):
    try:
        with open("../Data/%s.npz" % feature_set, 'r') as f:
            data = np.load(f)
            X, Xtest = data['X'], data['Xtest']
    except IOError:
            logging.warining("Could not find feature set %s", feature_set)
            return False
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
