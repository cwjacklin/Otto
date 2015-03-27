import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot, logsumexp, squared_norm
import ipdb
from scipy import optimize
import ipdb
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

logger = logging.getLogger(__name__)

def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or (n_classes * (n_features +
        1),)
            Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : ndarray, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w

def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or (n_classes * (n_features +
        1),)
            Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : ndarray, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)))
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def logistic_regression_path(X, y,  Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             copy=True, dual=False, penalty='l2',
                             bounds = None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Input data, target values.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    
    class_weight : {dict, 'auto'}, optional
        Over-/undersamples the samples of each class according to the given
        weights. If not given, all classes are supposed to have weight one.
        The 'auto' mode selects weights inversely proportional to class
        frequencies in the training set.
    
    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.
    Cs : ndarray
        Grid of Cs used for cross-validation.
    Notes
    -----
    You might get slighly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.
    """
    
    if type(Cs) is int:
        Cs = np.logspace(-4, 4, Cs)
    _, n_features = X.shape
    assert len(X) == len(y), "X and y must have the same length"
    
    classes = np.unique(y)
    lbin = LabelBinarizer()
    Y_bin = lbin.fit_transform(y)
    w0 = np.zeros((Y_bin.shape[1], n_features + int(fit_intercept)),
                                  order='F')
    w0 = w0.ravel()
    target = Y_bin
    n_vectors = classes.size

    sample_weight = np.ones(X.shape[0])
    func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
    coefs = list()
    for C in Cs:
        w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol,
                    maxiter = max_iter, bounds = bounds
                    )
        try:
            multi_w0 = np.reshape(w0, (classes.size, -1))
        except ValueError:
            ipdb.set_trace()
        coefs.append(multi_w0)
    return coefs, np.array(Cs)

class ConstrainedMultinomialClassifier(BaseEstimator):
    """ Constrained Multinomial Regression that used Scipy L-BFGS optimization
    
    This is similar to scikit-learn LogisticRegression, version .16. 
    The only addition is the ability to add interval constraints.
    It only supports L2 regularization.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.

    max_iter : int
        Maximum number of iterations for L-BFGS solver

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    tol : float, optional
        Tolerance for stopping criteria. (For the norm of gradient)

    verbose : int 
        Controls the frequency of output. 
        verbose < 1 means no output; 
        verbose == 1 means write messages to stdout; 
        verbose > 2 in addition means write logging information
        to a file named iterate.dat in the current working directory.
   
    bounds: list
        (min, max) pairs for each element in x, defining the bounds on that 
        parameter. Use None for one of min or max when there is no bound in 
        that direction.  
    
    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        Coefficient of the features in the decision function.

    intercept_ : array, shape (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.

    See also
    --------
    Scikit-learn LogisticRegression v.16  https://github.com/scikit-learn/
        scikit-learn/blob/master/sklearn/linear_model/logistic.py
    
    Scipy Optimize L-BFGS http://docs.scipy.org/doc/scipy/reference/generated/
        scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b
    """

    def __init__(self, tol = 1e-4, C = 1.0, fit_intercept = True, 
            random_state = None, max_iter = 100,
            bounds = None, verbose = 0):
        self.C = C
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.bounds = bounds

    def fit(self, X, y):
        """Fit the model according to the given training data.
        
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        self : object
            Returns self.
        """
        self.classes = np.unique(y) # np.unique sorts elements
        n_classes = len(self.classes)
        coef_, _ = logistic_regression_path(
                X, y, Cs = [self.C], 
                fit_intercept = self.fit_intercept, tol = self.tol, 
                verbose = self.verbose, max_iter = self.max_iter, 
                bounds = self.bounds)
        coef_ = coef_[0]
        if self.fit_intercept:
            self.intercept_ = coef_[:, -1]
            self.coef_ = coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)
            self.coef_ = coef_
        
        return self
    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        p = safe_sparse_dot(X, self.coef_.T)
        p += self.intercept_
        p -= logsumexp(p, axis=1)[:, np.newaxis]
        return np.exp(p, p)

    def predict(self, X):
        """Predict class labels for sample in X
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        p = self.predict_proba(X)
        idx = np.argmax(p, axis = 1)
        res = [self.classes[i] for i in idx]
        return np.array(res)

def GetBounds(n_yhats = 3, n_classes = 9, fit_intercept = True):
    I = np.identity(n_classes, dtype = int)
    I = np.hstack([I]*n_yhats)
    I = np.hstack([I, np.ones((n_classes, 1))])
    I = I.ravel()
    res = [(0, None) if i else (None, 0) for i in I]
    return res

def Test():
    """Testing ConstrainedMultinomialRegression
    
    Compare the results with scikit-learn LogisticRegression v.15
    
    Returns
    -------
    Log Loss for Logistic Regression, ConstrainedMultinomialRegression
    Accuracy for Logistic Regression, ConstrainedMultinomialRegression
    """
    n = 1000; p = 10; k = 3
    X = np.random.randn(n, p)
    beta = np.random.binomial(1, .5, (p, k))
    log_odd = X.dot(beta)
    prob = np.exp(log_odd)/(1 + np.exp(log_odd))
    y = np.array([np.argmax(i) for i in prob])
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    w = randn(k,p)
    cut = n/2
    train = np.arange(cut); valid = np.arange(cut,n) # Split Train and Test
    b = [(0,None)]*(p+1)*k # Constraint on Beta
    cl1 = LogisticRegression()
    cl2 = ConstrainedMultinomialClassifier(bounds = b)
    cl1.fit(X[train], y[train])
    cl2.fit(X[train], y[train])
    prob1 = cl1.predict_proba(X[valid])
    prob2 = cl2.predict_proba(X[valid])
    print log_loss(y[valid], prob1)
    print log_loss(y[valid], prob2)
    yhat1 = cl1.predict(X[valid])
    yhat2 = cl2.predict(X[valid])
    print accuracy_score(y[valid], yhat1)
    print accuracy_score(y[valid], yhat2)
