import numpy as np
import ipdb
import logging
from numpy.core.numeric     import identity
from numpy.linalg.linalg    import inv
from numpy.lib.twodim_base  import diag
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)
def DoubleExponential(x1, x2):
    return np.exp(-.5*cdist(x1, x2)**2)

def Matern52(x1, x2):
    d = cdist(x1, x2)
    return (1 + np.sqrt(5)*d + 5/3*d^2)*np.exp(-np.sqrt(5)*d)

def Matern32(x1, x2):
    d = cdist(x1, x2)
    return (1 + np.sqrt(3)*d)*np.exp(-np.sqrt(3)*d)

def f(x):
    return x*np.sin(np.pi*x)

def g(x, y):
    return -x**2 - y**2 + x*y +2*x + 2*y

def h(x, y, z):
    return x*np.sin(np.pi*x) + y*np.sin(y) + z*np.sin(z)

def GetGrid(n_params, n_grid, intv):
    res = np.empty((n_grid, n_params))
    for i in xrange(n_params):
        res[:,i] = np.random.uniform(low = intv[0], high = intv[1], 
                                    size = n_grid) 
    return np.matrix(res)

def GPUCB(func = f, n_params = 1, kernel = DoubleExponential, intv = [0,5],
          sig = .1, mu_prior = 0, sigma_prior = 1, n_grid = 100, n_iter = 10):
    np.random.seed(1)
    grid = GetGrid(n_params, n_grid, intv)
    mu = np.zeros(n_grid) + mu_prior
    sigma = np.ones(n_grid)*sigma_prior
    X = np.matrix(np.empty((n_iter, n_params)))
    y = np.empty(n_iter)
    logger.info("%4s |%6s |%6s", "Iter", "Func", "Max")
    for i in xrange(n_iter):
        beta = 2*np.log((i+1)**2*2*np.pi**2/3/.1) + \
               2*n_params*np.log((i+1)**2*n_params*(intv[1] - intv[0]))
        idx = np.argmax(mu + np.sqrt(beta)*sigma)
        X[i,:] = grid[idx]
        logger.info(X[i])
        y[i] = func(*np.array(X)[i])
        invKT = inv(kernel(X[:i+1], X[:i+1])*sigma_prior**2 + 
                    sig**2*identity(i + 1))
        grid = GetGrid(n_params, n_grid, intv)
        kT = kernel(X[:i+1], grid)*sigma_prior**2
        mu = mu_prior + kT.T.dot(invKT).dot(y[:i+1] - mu_prior)
        sigma2 = np.ones(n_grid)*sigma_prior**2 - diag(kT.T.dot(invKT).dot(kT))
        sigma = np.sqrt(sigma2)
        logger.info("%4d |%6.2f |%6.2f", i, y[i], np.max(y[:i+1]))
    return {'X': X,
            'y': y,
            'mu': mu,
            'beta': beta,
            'sigma': sigma,
            'grid': grid}
