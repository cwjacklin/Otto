from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import log_loss
from sklearn.metrics      import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import numpy as np

### 1. Feature Engineering
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

### 2.
def getGrid(center, length = 5, scale = 2):
    return center*(scale + 0.)**np.arange(-length, length)

def logLossAdj(y, yhat, inflate = 0.001):
    return log_loss(y, (yhat + inflate)/(1 + 9*inflate))

def logLossAdjGrid(y, yhat):
    return np.min([logLossAdj(y, yhat, eps = inflate) for inflate in
                       getGrid(0.0001, 10)])

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
