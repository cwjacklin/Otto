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
