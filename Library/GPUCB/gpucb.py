import numpy as np
import ipdb
import logging
import time
from numpy.core.numeric     import identity
from numpy.linalg.linalg    import inv
from numpy.lib.twodim_base  import diag
from scipy.spatial.distance import cdist
from scipy.stats            import uniform, randint
from sklearn.base           import BaseEstimator
logger = logging.getLogger(__name__)
### 1. Kernels
def DoubleExponential(x1, x2, rho = 1./5):
    return np.exp(-.5*cdist(x1, x2)**2/x1.shape[1]/rho**2)

def DoubleExponentialW(invrho = 5.):
    return lambda x1, x2: DoubleExponential(x1, x2, rho = 1./invrho)

def Matern52(x1, x2, rho = 1./5):
    d = cdist(x1, x2)/np.sqrt(x1.shape[1])/rho
    return (1 + np.sqrt(5)*d + 5/3*d**2)*np.exp(-np.sqrt(5)*d)

def Matern32(x1, x2, rho = 1./10):
    d = cdist(x1, x2)/np.sqrt(x1.shape[1])/rho
    return (1 + np.sqrt(3)*d)*np.exp(-np.sqrt(3)*d)

def Matern32W(invrho = 10):
    return lambda x1, x2: Matern32(x1, x2, rho = 1./invrho)
### 2. Sample Functions to Maximize
def f(x):
    return x*np.sin(np.pi*1*x)

def g(x, y):
    return -x**2 - y**2 + x*y +2*x + 2*y

def g2(x, y):
    return x*np.sin(np.pi*x) + y*np.sin(np.pi*y)

def h(x, y, z):
    return x*np.sin(np.pi*x) + y*np.sin(y) + z*np.sin(z)


### 3. Distribution to Sample the Parameters from
class Uniform():
    def __init__(self, start = 0, end = 1):
        self.start = start
        self.end = end
    def rvs(self, size = 1):
        return uniform(loc = self.start, scale = self.end - 
                       self.start).rvs(size)
    def scale(self):
        return self.end - self.start

class UniformInt():
    def __init__(self, start = 0, end = 2):
        self.start = start
        self.end = end
    def rvs(self, size = 1):
        return randint(low = self.start, high = self.end).rvs(size)
    def scale(self):
        return self.end - self.start

class LogUniform():
    def __init__(self, start = .1, end = 10):
        self.start = start
        self.end   = end
    def rvs(self, size = 1):
        res = uniform(loc = np.log10(self.start), 
                scale = np.log10(self.end) - np.log10(self.start)).rvs(size)
        return 10**res
    def scale(self):
        return np.log10(self.end) - np.log10(self.start)
        
class LogUniformInt():
    def __init__(self, start = 1, end = 100):
        self.start = start
        self.end = end
    def rvs(self, size = 1):
        res = LogUniform(start = self.start, end = self.end).rvs(size)
        return np.array(res, dtype = int)
    def scale(self):
        return np.log10(self.end) - np.log10(self.start)

### 4. Gaussian Process UCB
def GetRandGrid(n_grid, params_dist):
    n_params = len(params_dist)
    grid        = np.empty((n_grid, n_params))
    grid_scaled = np.empty((n_grid, n_params))
    for (i, param) in enumerate(params_dist.keys()):
        distribution = params_dist[param]
        grid[:,i] = distribution.rvs(size = n_grid)
        if distribution.__class__.__name__.startswith("Log"):
            grid_scaled[:,i] = np.log10(grid[:,i])/distribution.scale()
        else:
            grid_scaled[:,i] = grid[:,i]/distribution.scale()
    return grid, np.matrix(grid_scaled)

def GPUCB(func = f, kernel = DoubleExponential,
        params_dist = {'x': Uniform(start = 0, end = 5)},
        prev_X = None, prev_y = None,
          sig = .1, mu_prior = 0, sigma_prior = 1, n_grid = 100, n_iter = 10,
          seed = 2, time_budget = 36000):
    time_start = time.time()
    np.random.seed(seed)
    n_params          = len(params_dist)
    params_name       = params_dist.keys()
    grid, grid_scaled = GetRandGrid(n_grid, params_dist)
    mu = np.zeros(n_grid) + mu_prior
    sigma = np.ones(n_grid)*sigma_prior
    X = np.empty((n_iter, n_params))
    X_scaled = np.matrix(np.empty((n_iter, n_params)))
    y = np.empty(n_iter)
    logger.info("%4s |%9s |%9s |%s", "Iter", "Func", "Max",
            '|'.join(['{:6s}'.format(i) for i in params_name]))
    for i in xrange(n_iter):
        #beta = 2*np.log((i+1)**2*2*np.pi**2/3/.1) + \
        #       2*n_params*np.log((i+1)**2*n_params)
        beta = (i+1)**2
        #ipdb.set_trace()
        idx = np.argmax(mu + np.sqrt(beta)*sigma)
        X[i,:] = grid[idx]
        X_scaled[i] = grid_scaled[idx]
        y[i] = func(**dict(zip(params_name, X[i])))
        invKT = inv(kernel(X_scaled[:i+1], X_scaled[:i+1])*sigma_prior**2 + 
                    sig**2*identity(i + 1))
        grid, grid_scaled = GetRandGrid(n_grid, params_dist)
        kT = kernel(X_scaled[:i+1], grid_scaled)*sigma_prior**2
        mu = mu_prior + kT.T.dot(invKT).dot(y[:i+1] - mu_prior)
        sigma2 = np.ones(n_grid)*sigma_prior**2 - diag(kT.T.dot(invKT).dot(kT))
        sigma = np.sqrt(sigma2)

        logger.info("%4d |%9.4g |%9.4g |%s" , i, y[i], np.max(y[:i+1]),
                '|'.join(['{:6.2g}'.format(i) for i in X[i]]))
        if time.time() - time_start > time_budget:
            break
        ipdb.set_trace()
        if True:
            figure(1); plt.clf(); xlim((0,5)); ylim(-4,10);
            index = np.argsort(grid[:,0])
            gr = grid[:,0]
            plot(gr[index], mu[index], color = 'red', label = "Mean")
            plot(gr[index], mu[index] + sigma[index], color = 'blue', 
                    label = "Mean + Sigma")
            plot(gr[index], mu[index] - sigma[index], color = 'blue',
                    label = "Mean - Sigma")
            plot(X[:i+1,0], y[:i+1], 'o', color = 'green', label = "Eval Points")
            plot(np.linspace(0,5, num = 500),func(np.linspace(0,5, num = 500)),
                    color = 'green', label = "True Func")
            plot(gr[index], mu[index] + sqrt(beta)*sigma[index], 
                    color = 'yellow', label = "Mean + sqrt(B)*Sigma")
            plt.grid()
            legend(loc = 2)
            show()
    return {'X': X,
            'y': y,
            'mu': mu,
            'beta': beta,
            'sigma': sigma,
            'grid': grid}

class GPUCBOpt(BaseEstimator):
    """ Scikit-learn inspired Gaussian Process Upper Confidence Bound 
    Optimization of hyperparameters (blackbox functions)
    It automatically rescale all the hyperparameter range to [0,1], this is to
    avoid d parameters going into Kernel (e.g. DoubleExponential) that we
    usually need to scale the different axis appropriately.
    
    It currently support Uniform and LogUniform distribution for hyperparameter
    to sample from (similar to RandomizedSearchCV of scikit-learn)
    
    We do maximization through out this class. You can minimize f(x) by maximize
    -f(x)
    Parameters
    ----------
    kernel: function
        The Kernel for Gaussian Process. Support DoubleExponential, 
        Matern32, and Matern52

    sig: float, optional
        Similar to the lambda in Ridge regression, a term to add to the diagonal
        of the covariance matrix. It acts as a regularizer. The other
        interpretation is the variance of function evaluation, which means we
        can only observe y = f(x) + e, for e zero mean Gaussian, variance sig**2
    
    mu_prior: float
        This is the mean prior of function value, as modelled by a Gaussian
        Process. Put in your estimate.
    
    sigma_prior: float
        This is the standard deviation prior of function, as modeled by a
        Gaussian Process.

    n_grid: int
        At each iteration, n_grid point is sample to pick the best point to
        evaluate the function (the argmax part in GP-UCB algorithm). More n_grid
        point is better, at the expense of computational time. 

    max_iter: int
        Maximum iterations run

    random_state: int
        The random state for numpy seed

    time_budget: int (second)
        Iterations will stop if total time run exceeds this number,
        
    verbose: bool
        Whether to print progress or not
    """
    def __init__(self, sig = .1, mu_prior = 0., sigma_prior = 1., n_grid = 100,
            max_iter = 10, random_state = None, time_budget = 36000,
            verbose = 1, kernel = DoubleExponential,
            file_id = "111"):
        self.sig            = sig
        self.mu_prior       = mu_prior
        self.sigma_prior    = sigma_prior
        self.n_grid         = n_grid
        self.max_iter       = max_iter
        self.random_state   = random_state
        self.time_budget    = time_budget
        self.verbose        = verbose
        self.kernel         = kernel
        self.file_id        = file_id

    def get_grid(self, params_dist):
        self.n_params = len(params_dist)
        grid        = np.empty((self.n_grid, self.n_params))
        grid_scaled = np.empty((self.n_grid, self.n_params))
        self.params_name = []
        for (i, param) in enumerate(params_dist.keys()):
            self.params_name.append(param)
            distribution = params_dist[param]
            grid[:,i] = distribution.rvs(size = self.n_grid)
            if distribution.__class__.__name__.startswith("Log"):
                grid_scaled[:,i] = np.log10(grid[:,i])/distribution.scale()
            else:
                grid_scaled[:,i] = grid[:,i]/distribution.scale()
        return grid, grid_scaled

    def fit(self, func, params_dist, pre_X = None, pre_y = None):
        time_start = time.time()
        if self.random_state is not None: np.random.seed(self.random_state)
        grid, grid_scaled = self.get_grid(params_dist)
        mu       = np.zeros(self.n_grid) + self.mu_prior
        sigma    = np.ones(self.n_grid)*self.sigma_prior
        X        = np.zeros((self.max_iter, self.n_params))
        X_scaled = np.matrix(np.zeros((self.max_iter, self.n_params)))
        y        = np.zeros(self.max_iter)
        if (pre_X is not None) and (pre_y is not None):
            pre_X_mat, pre_X_scaled = self.scale(pre_X, pre_y, params_dist)
            X        = np.vstack([pre_X_mat, X]) 
            X_scaled = np.vstack([pre_X_scaled, X_scaled])
            y        = np.concatenate([pre_y, y])
            pre_len  = len(pre_y)
        else: pre_len = 0
        if self.verbose:
            logger.info('%4s|%9s|%9s|%9s', 'Iter','Func','Max',
                    '|'.join(['{:9s}'.format(i) for i in self.params_name]))
        for i in xrange(pre_len, pre_len + self.max_iter):
            beta        = (i + 1)**2
            idx         = np.argmax(mu + np.sqrt(beta)*sigma)
            X[i,:]      = grid[idx]
            X_scaled[i] = grid_scaled[idx]
            y[i]        = func(**dict(zip(self.params_name, X[i])))
            KT          = self.kernel(X_scaled[:(i + 1)], X_scaled[:(i + 1)])*\
                            self.sigma_prior
            invKT       = inv(KT + self.sig**2*identity(i + 1))
            grid, grid_scaled = self.get_grid(params_dist)
            kT          = self.kernel(X_scaled[:(i + 1)], grid_scaled)*\
                          self.sigma_prior**2
            mu          = self.mu_prior + \
                          kT.T.dot(invKT).dot(y[:(i + 1)] - self.mu_prior)
            sigma2      = np.ones(self.n_grid)*self.sigma_prior**2 - \
                            diag(kT.T.dot(invKT).dot(kT))
            sigma       = np.sqrt(sigma2)
            ### Save Data
            np.savez_compressed(func.__name__ + self.file_id + ".npz", 
                    X = X[:(i+1)], y = y[:(i+1)], mu = mu, grid = grid,
                    sigma = sigma)
            if self.verbose:
                logger.info('%4d|%9.4g|%9.4g|%s', i, y[i], np.max(y[:(i + 1)]),
                        '|'.join(['{:9.4g}'.format(i) for i in X[i]]))
            if time.time() - time_start > self.time_budget:
                break
        self.X      = X[:(i + 1)]
        self.y      = y[:(i + 1)]
        self.mu     = mu
        self.beta   = beta
        self.sigma  = sigma
        self.grid   = grid

    def scale(self, pre_X, pre_y, params_dist):
        pre_X_mat    = np.zeros((len(pre_y), self.n_params)) 
        pre_X_scaled = np.zeros((len(pre_y), self.n_params))
        for (i, param) in enumerate(self.params_name):
            pre_X_mat[:,i] = pre_X[param]
            distribution = params_dist[param]
            if distribution.__class__.__name__.startswith("Log"):
                pre_X_scaled[:,i] = np.log10(pre_X[param])/distribution.scale()
            else:
                pre_X_scaled[:,i] = np.array(pre_X[param])/distribution.scale()
        return pre_X_mat, pre_X_scaled
