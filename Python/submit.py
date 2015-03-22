from multilayer_perceptron import *
from sklearn.neighbors import KNeighborsClassifier
def SubmitSVM():
    X, Xtest = TextTransform(X, Xtest)
    svm = SVC(C = 4., cache_size = 200, class_weight = None, coef0 = 0.0, 
              degree = 3, gamma = 2.0, kernel = 'rbf', max_iter = -1, 
              probability = True, random_state = None, 
              shrinking = True, tol = 0.001, verbose = True)
    svm.fit(X,y)
    yhat = svm.predict_proba(X)
    yhattest = svm.predict_proba(Xtest)
    getSubmission("svm.csv", yhat)

def SubmitNN(X, Xtest):
    X, Xtest = TextTransform(X, Xtest)
    epsilon = 2e-5
    write(str('epsilon'))
    nn = MultilayerPerceptronClassifier(
            hidden_layer_sizes = 320, max_iter = 200, 
            alpha = 6.4e-5, activation = 'relu', verbose = True)
    nn.fit(X, y)
    write(str(nn.score(X, y)))
    yhat = nn.predict_proba(Xtest)
    getSubmission("nn.csv", yhat, eps = epsilon)


def SubmitkNN(X, Xtest):
    X, Xtest = TextTransform(X, Xtest)
    epsilon = 1e-4
    knn = KNeighborsClassifier(n_neighbors = 200, weights = 'uniform', 
            p = 1)
    knn.fit(X, y)
    write(str(knn.score(X, y)))
    yhat = knn.predict_proba(Xtest)
    getSubmission("knn.csv", yhat, eps = epsilon)

SubmitkNN(X, Xtest)
