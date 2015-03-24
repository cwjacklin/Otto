from sklearn.metrics      import log_loss
from sklearn.metrics      import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import numpy as np
import pandas as pd
from re import sub

### 1. Feature Engineering
def Write(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def TextTransform(X, Xtest = None):
    Write("Process Data with TFIDF...\n")
    tfidf = TfidfTransformer()
    if Xtest is None:
        X = tfidf.fit_transform(X).toarray()
        return X
    else:
        tfidf.fit(X)
        return tfidf.transform(X).toarray(), tfidf.transform(Xtest).toarray()

def LogTransform(X):
    Write("Process Data with Log10...\n")
    return np.log10(X)

def AddSquare(X):
    Write("Adding Quadratic Terms..\n")
    return np.hstack([X,X**2])

### 2.
def GetGrid(center, length, mode = 'mul', scale = 2):
    if mode == 'mul':
        return center*(scale + 0.)**np.arange(-length, length)
    elif mode == 'add':
        return center + scale*np.arange(-length, length)

def LogLossAdj(y, yhat, inflate = 0.001):
    return log_loss(y, (yhat + inflate)/(1 + 9*inflate))

def LogLossAdjGrid(y, yhat):
    grid = GetGrid(0.0001, 10)
    res = [log_loss(y, yhat, eps = inflate) for inflate in grid]
    Write("Optimal Logloss: %.4f, for Eps: %.4g \n" % (
        min(res), grid[np.argmin(res)]))
    return np.min(res)

def GetSubmission(file_name, yhat, eps):
    submission = pd.read_csv('../Data/sampleSubmission.csv')
    yhat = np.maximum(eps, yhat)
    yhat = np.minimum(1 - eps, yhat)
    yhat = pd.DataFrame(yhat, index = submission.id.values,
            columns = submission.columns[1:])
    assert (len(yhat) == len(submission))
    yhat.to_csv(file_name, index_label = 'id')

def GetSubmission2(file_name, model, X, y, Xtest, **kwargs):
    time_before = time.time()
    md = model(**kwargs)
    Write("Start Training " + str(model).split(" ")[-1].split(".")[-1][:-2] + \
                      "\n" + str(kwargs) + "\n")
    md.fit(X, y)
    yhat = md.predict_proba(Xtest)
    yhat = (yhat + 0.001)/1.009
    submission = pd.read_csv('../Data/sampleSubmission.csv')
    yhat = pd.DataFrame(yhat, index = submission.id.values,
                                    columns = submission.columns[1:])
    yhat.to_csv(file_name, index_label = 'id')

def TransformLabel(y):
    res = np.empty(shape = (len(y), 9))
    for i in range(9):
        res[:,i] = (y == 'Class_' + str(i+1)) + 0
    return res

def stringify(model, feature_set):
    """Given a model and a feature set, return a short string that will serve
    as identifier for this combination.
    Ex: (LogisticRegression(), "basic_s") -> "LR:basic_s"
    """
    return "%s:%s" % (sub("[a-z]", '', model.__class__.__name__), feature_set)
