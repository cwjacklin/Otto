import numpy as np
from numpy.random import randn
from scipy.stats import truncnorm
from gpucb import Uniform, UniformInt, LogUniform, LogUniformInt
import logging
if __name__ == '__main__':
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
            filename="history.log",
            filemode='a', level=logging.DEBUG,
            datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
            datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)

np.random.seed(2)
def GetSample(param_ranges):
    res = {}
    for key in param_ranges.keys():
        r = param_ranges[key]
        res[key] = Uniform(r[0], r[1]).rvs()[0]
    return res

def GetDirection(x, scale, param_ranges):
    res = {}
    for key in param_ranges.keys():
        r = param_ranges[key]
        x0 = x[key]
        direction = truncnorm(a = (r[0] - x0)/(r[1] - r[0])/scale, 
                b = (r[1] - x0)/(r[1] - r[0])/scale).rvs(1)[0]
        direction = direction*scale*(r[1] - r[0])
        res[key] = x0 + direction
    return res

class RandomDirection():
    def __init__(self, decay_start, decay_stop, max_iter = 20):
        self.max_iter = max_iter
        self.decays = np.logspace(decay_start, decay_stop, num = max_iter)
    def fit(self, func, param_ranges):
        x = GetSample(param_ranges)
        y = func(**x)
        logger.info('%4s|%9s|%9s|%10s','Iter', 'Func', 'Max', 
                '|'.join(['{:6s}'.format(i) for i in param_ranges.keys()]))
        for i in xrange(1, self.max_iter):
            x_new = GetDirection(x, self.decays[i], param_ranges)
            y_new = func(**x_new)
            logger.info('%4d|%9.4g|%9.4g|%s', i, y_new, max(y, y_new),
                    '|'.join(['{:6.2g}'.format(i) for i in x_new.values()]))
            if y_new > y:
                x = x_new; y = y_new
        self.best = max(y, y_new)

def f(x, y):
    return x*np.sin(np.pi*x) + y*np.sin(np.pi*y)

clf = RandomDirection(decay_start = 0, decay_stop = -3, max_iter = 100)
clf.fit(func = f, param_ranges = {'x': [-5,0], 'y': [0,5]})


if False:
    x0 = 2
    r = [1, 10]
    scale = .005
    direction = truncnorm(a = (r[0] - x0 + .0)/(r[1] - r[0])/scale, 
                    b = (r[1] - x0 + .0)/(r[1] - r[0])/scale).rvs(1000)*scale*\
                    (r[1] - r[0])
    figure(1); plt.clf(); hist(x0 + direction, bins = 20)
    plt.show()
