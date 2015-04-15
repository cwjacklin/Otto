import numpy as np
import ipdb
import logging
import time
from numpy.core.numeric     import identity
from numpy.linalg.linalg    import inv
from numpy.lib.twodim_base  import diag
from scipy.spatial.distance import cdist
from scipy.stats            import uniform, randint

logger = logging.getLogger(__name__)
### 1. Kernels
def DoubleExponential(x1, x2, rho = 1./5):
    return np.exp(-.5*cdist(x1, x2)**2/x1.shape[1]/rho**2)

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
    return x*np.sin(np.pi*6*x)

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
    for (i, distribution) in enumerate(params_dist.values()):
        grid[:,i] = distribution.rvs(size = n_grid)
        if distribution.__class__.__name__.startswith("Log"):
            grid_scaled[:,i] = np.log10(grid[:,i])/distribution.scale()
        else:
            grid_scaled[:,i] = grid[:,i]/distribution.scale()
    return grid, np.matrix(grid_scaled)

def GPUCB(func = f, kernel = DoubleExponential,
        params_dist = {'x': Uniform(start = 0, end = 5)},
          sig = .1, mu_prior = 0, sigma_prior = 1, n_grid = 100, n_iter = 10,
          seed = 2, time_budget = 36000):
    time_start = time.time()
    np.random.seed(seed)
    n_params = len(params_dist)
    params_name = params_dist.keys()
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
        #if False:
        #    figure(1); clf(); xlim((0,5)); ylim(-4,10);
        #    index = np.argsort(grid[:,0])
        #    gr = grid[:,0]
        #    plot(gr[index], mu[index], color = 'red', label = "Mean")
        #    plot(gr[index], mu[index] + sigma[index], color = 'blue', 
        #            label = "Mean + Sigma")
        #    plot(gr[index], mu[index] - sigma[index], color = 'blue',
        #            label = "Mean - Sigma")
        #    plot(X[:i+1,0], y[:i+1], 'o', color = 'green', label = "Eval Points")
        #    plot(np.linspace(0,5, num = 500),func(np.linspace(0,5, num = 500)),
        #            color = 'green', label = "True Func")
        #    plot(gr[index], mu[index] + sqrt(beta)*sigma[index], 
        #            color = 'yellow', label = "Mean + sqrt(B)*Sigma")
        #    plt.grid()
        #    legend(loc = 2)
        #    show()
    return {'X': X,
            'y': y,
            'mu': mu,
            'beta': beta,
            'sigma': sigma,
            'grid': grid}
