from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss
from sklearn.metrics      import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search  import GridSearchCV
import sys
import numpy as np

def write(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def TextTransform(X, train = None, valid = None):
    write("Process Data with TFIDF...\n")
    tfidf = TfidfTransformer()
    if (train == None):
        X = tfidf.fit_transform(X).toarray()
        return X
    else:
        tfidf.fit(X[train])
        return tfidf.transform(X).toarray()

def LogTransform(X):
    write("Process Data with Log10...\n")
    return np.log10(X)

def AddSquare(X):
    write("Adding Quadratic Terms..\n")
    return np.hstack([X,X**2])

if __name__ == "__main_":
    LoadData()
    X = TextTransform(X, train, valid)
    #X = AddSquare(X)

    time_before = time.time()
    model = RandomForestClassifier(n_estimators = 96, n_jobs = 24, verbose = True,)
    model.fit(X[train], y[train])
    yhat = model.predict_proba(X[valid])
    print log_loss(y[valid], (yhat+0.001)/1.009)
    print("Running Time: " + str(time.time() - time_before) + "\n")

    time_before = time.time()
    model = LogisticRegression(penalty = 'l1', C = 12)
    params = dict(C = [0.25, 0.5, 1, 2, 4, 8])
    scorer = make_scorer(log_loss, greater_is_better = False, needs_proba = True)
    grid = GridSearchCV(model, params, score, cv = 5, verbose = 2, n_jobs = 10)
    grid.fit(X[train], y[train])
    yhat = model.predict_proba(X[valid])
    print log_loss(y[valid], yhat)
    print("Running Time: " + str(time.time() - time_before) + "\n")

