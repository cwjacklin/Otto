from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
import numpy as np
import theano
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from data import GetDataset, LoadData
from ml import *

if True:
    X, Xtest = GetDataset("original")
    X = X.astype('float32')
    _, y, _ = LoadData(); del _
    Y = np.array([int(i[-1]) for i in y])
    Y = Y.astype('int32') - 1

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start = 0.03, stop = 0.001, is_log = True):
        self.name = name
        self.is_log = is_log
        if is_log:
            self.start, self.stop = np.log10(start), np.log10(stop)
        else: 
            self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            if self.is_log:
                self.ls = np.logspace(self.start, self.stop,
                                        nn.max_epochs)
            else:
                self.ls = np.linspace(self.start, self.stop,
                                        nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                        self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

l_start = .017
l_stop = 1e-6
m_start = .9
m_stop = .999
max_epochs = 3000
print l_start, l_stop, m_start, m_stop, max_epochs
print "DDD"
params = dict(
    layers = [
        ('input', layers.InputLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape = (None, 186),
    dropout1_p = .1,
    hidden1_num_units = 1500,
    dropout2_p = .7,
    hidden2_num_units = 1500,
    dropout3_p = .7,
    hidden3_num_units = 1500,
    dropout4_p = .7,
    output_nonlinearity = softmax,
    output_num_units = 9,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(l_start)),
    update_momentum = theano.shared(float32(m_start)),

    regression = False,
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = l_start, 
            stop = l_stop, is_log = True),
        AdjustVariable('update_momentum', start = m_start, 
            stop = m_stop, is_log = False),
        ],
    max_epochs = max_epochs,
    verbose = 1,
    )

def OptNN2(d0, h, d1, m_ep):
    h = int(h); m_ep = int(m_ep)
    logger.info("Params:..%f, %d, %f, %d", d0, h, d1, m_ep)
    params = dict(
        layers = [
            ('input', layers.InputLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden3', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],

        input_shape = (None, 93),
        dropout1_p = d0,
        hidden1_num_units = h,
        dropout2_p = d1,
        hidden2_num_units = h,
        dropout3_p = d1,
        hidden3_num_units = h,
        dropout4_p = d1,
        output_nonlinearity = softmax,
        output_num_units = 9,

        update = nesterov_momentum,
        update_learning_rate = theano.shared(float32(l_start)),
        update_momentum = theano.shared(float32(m_start)),

        regression = False,
        on_epoch_finished = [
            AdjustVariable('update_learning_rate', start = l_start, 
                stop = l_stop, is_log = True),
            AdjustVariable('update_momentum', start = m_start, 
                stop = m_stop, is_log = False),
            ],
        max_epochs = m_ep,
        verbose = 1,
        )

    CVScores = []
    res = np.empty((len(Y), len(np.unique(Y))))
    kcv = StratifiedKFold(Y, 5, shuffle = True); i = 1
    for train_idx, valid_idx in kcv:
        logger.info("Running fold %d...", i); i += 1
        net = NeuralNet(**params)
        net.set_params(eval_size = None)
        net.fit(X[train_idx], Y[train_idx])
        res[valid_idx, :] = net.predict_proba(X[valid_idx])
        CVScores.append(log_loss(Y[valid_idx], res[valid_idx]))
    return -np.mean(CVScores)

def OptNN(d1, h1, d2, h2, d3, start, stop, max_epochs):
    params2 = params.copy()
    on_epoch = [AdjustVariable('update_learning_rate', 
                               start = start, stop = stop),
                AdjustVariable('update_momentum', start = .9, stop = .999)]
    params2['dropout1_p']           = d1
    params2['dropout2_p']           = d2
    params2['dropout3_p']           = d3
    params2['dropout4_p']           = d4
    params2['hidden1_num_units']    = h1
    params2['hidden2_num_units']    = h2
    params2['hidden3_num_units']    = h3
    params2['max_epochs']           = max_epochs
    params2['on_epoch_finished'] = on_epoch
    kcv = StratifiedKFold(Y, 5, shuffle = True)
    res = np.empty((len(Y), len(np.unique(Y)))); i = 1
    CVScores = []
    for train_idx, valid_idx in kcv:
        logger.info("Running fold %d...", i); i += 1
        net = NeuralNet(**params2)
        net.set_params(eval_size = None)
        net.fit(X[train_idx], Y[train_idx])
        res[valid_idx, :] = net.predict_proba(X[valid_idx]) 
        CVScores.append(log_loss(Y[valid_idx], res[valid_idx]))
    return -np.mean(CVScores)

if __name__ == '__main__':
    if False:
        score = OptNN2(.1, 512, .4, 10)
        clf = GPUCBOpt(kernel = Matern32W(invrho = 10), max_iter = 1000,
                     mu_prior = -.63, sigma_prior = .10, sig = .005,
                       n_grid = 1000, random_state = int(job_id),
                  time_budget = 3600*int(job_id), verbose = 1)
        params_dist = {'d0': Uniform(0, .4),
                        'd1': Uniform(.1, .6),
                        'h' : UniformInt(200, 1000),
                        'm_ep': UniformInt(300,3000)
                       }  
        clf.fit(OptNN2, params_dist)
    res = np.zeros(1000)
    for i in xrange(1000):
        d0 = np.random.uniform(.1,.2)
        d1 = np.random.uniform(.3,.6)
        h  = np.random.randint(500,1200)
        m_ep = np.random.randint(300, 3000)
        res[i] = OptNN2(d0 = d0, h = h, d1 = d1, m_ep = m_ep)
        logger.info("LogLoss: %10.4f ", res[i])
        np.savez_compressed("res2_NN.npz", res = res)
