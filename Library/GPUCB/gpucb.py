import numpy as np
import ipdb
from numpy.core.numeric     import identity
from numpy.linalg.linalg    import inv
from numpy.lib.twodim_base  import diag
from scipy.spatial.distance import cdist

def DoubleExponential(x1, x2):
    return np.exp(-.5*cdist(x1, x2)**2)

def f(x):
    return x*sin(pi*x)

def g(x, y):
    return -x**2 - y**2 + x*y +2*x + 2*y

def h(x, y, z):
    return x*sin(pi*x) + y*sin(y) + z*sin(z)

def GetGrid(n_params, n_grid, intv):
    res = np.empty((n_grid, n_params))
    for i in xrange(n_params):
        res[:,i] = np.random.uniform(low = intv[0], high = intv[1], 
                                    size = n_grid) 
    return np.matrix(res)

def GPUCB(func = f, n_params = 1, kernel = DoubleExponential, intv = [0,5],
          sig = .1, n_grid = 100, n_iter = 10):
    np.random.seed(314)
    grid = GetGrid(n_params, n_grid, intv)
    mu = np.zeros(n_grid)
    sigma = np.ones(n_grid)
    X = np.matrix(np.empty((n_iter, n_params)))
    y = np.empty(n_iter)
    logger.info("%4s |%6s |%6s", "Iter", "Func", "Max")
    for i in xrange(n_iter):
        # ipdb.set_trace()
        beta = 2*np.log((i+1)**2*2*pi**2/3/.1) + \
               2*n_params*np.log((i+1)**2*n_params*(intv[1] - intv[0]))
        idx = np.argmax(mu + sqrt(beta*sigma)RRSRss
        X[i,:] = grid[idx]
        y[i] = func(*np.array(X)[i])
        invKT = inv(kernel(X[:i+1], X[:i+1]) + sig**2*identity(i + 1))
        grid = GetGrid(n_params, n_grid, intv)
        kT = kernel(X[:i+1], grid)
        mu = kT.T.dot(invKT).dot(y[:i+1])
        sigma = np.ones(n_grid) - diag(kT.T.dot(invKT).dot(kT))
        logger.info("%4d |%6.2f |%6.2f", i, y[i], np.max(y[:i+1]))
    return X, y
