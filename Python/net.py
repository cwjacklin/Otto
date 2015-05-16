from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
import numpy as np
import theano
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from ml import *

if False:
    X, Xtest = GetDataset("ensemble", 
            ensemble_list = ['btc','btc2','btc3','btc4','svc','svc2','svc3',
                'nn','nn2','nic', 'mpc','knc','etc','cccv', 'log'])
    X, Xtest = GetDataset('text-standardized')
    #X = X.astype('float32')
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
        EarlyStopping(patience=200),
        ],
    max_epochs = max_epochs,
    verbose = 1,
    )

def OptNN2(d0, d1,d2, d3, h1, h2, h3, me, ls, le):
    h1, h2, h3 = int(h1), int(h2), int(h3); 
    me = int(me)
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
        hidden1_num_units = h1,
        dropout2_p = d1,
        hidden2_num_units = h2,
        dropout3_p = d2,
        hidden3_num_units = h3,
        dropout4_p = d3,
        output_nonlinearity = softmax,
        output_num_units = 9,

        update = nesterov_momentum,
        update_learning_rate = theano.shared(float32(l_start)),
        update_momentum = theano.shared(float32(m_start)),

        regression = False,
        on_epoch_finished = [
            AdjustVariable('update_learning_rate', start = ls, 
                stop = le, is_log = True),
            AdjustVariable('update_momentum', start = m_start, 
                stop = m_stop, is_log = False),
            ],
        max_epochs = me,
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

def OptMPC(max_iter, hidden_layer_sizes, alpha, learning_rate_init, power_t,
        learning_rate):
    if learning_rate: learning_rate = 'invscaling'
    else:             learning_rate = 'constant'
    kcv = StratifiedKFold(y, 5, shuffle = True)
    res = np.empty((len(y), len(np.unique(y))))
    for train_idx, valid_idx in kcv:
        clf = MultilayerPerceptronClassifier(max_iter = max_iter,
            hidden_layer_sizes  = hidden_layer_sizes,
            alpha               = alpha,
            learning_rate_init  = learning_rate_init,
            power_t             = power_t)
        clf.fit(X[train_idx], y[train_idx])
        res[valid_idx] = clf.predict_proba(X[valid_idx])
    return -log_loss(y, res)


if False:
    params_dist = {
            'max_iter'          : UniformInt(100, 1000),
            'hidden_layer_sizes': UniformInt(100, 1000),
            'alpha'             : LogUniform(2**-15, 1),
            'learning_rate'     : UniformInt(0,2),
            'learning_rate_init': LogUniform(2**-10, 1),
            'power_t'           : Uniform(.3, .99)
            }
    clf = GPUCBOpt(kernel = DoubleExponential, max_iter = 100, 
            mu_prior = -.50, sigma_prior = .10, sig = .005, 
            n_grid   = 1000, random_state = int(job_id),
            time_budget = 3600*12*7, verbose = 1)
    clf.fit(OptMPC, params_dist)

if False:
    clf = GPUCBOpt(kernel = DoubleExponential, max_iter = 1000,
		 mu_prior = -.63, sigma_prior = .10, sig = .005,
		   n_grid = 1000, random_state = None,
	      time_budget = 3600*172, verbose = 1)
    params_dist = { 'd0': Uniform(.0, .4),
		    'd1': Uniform(.0, .6),
		    'd2': Uniform(.0, .6),
		    'd3': Uniform(.0, .6),
		    'h1': UniformInt(200, 2000),
		    'h2': UniformInt(200, 2000),
		    'h3': UniformInt(200, 2000),
		    'me': UniformInt(300,3000),
		    'ls': LogUniform(.001, .1),
		    'le': LogUniform(1e-8, 1e-4),
		   }  
    clf.fit(OptNN2, params_dist)
    """
    res = np.zeros(1000)
    for i in xrange(1000):
        d0 = np.random.uniform(.1,.2)
        d1 = np.random.uniform(.3,.6)
        h  = np.random.randint(500,1200)
        m_ep = np.random.randint(300, 3000)
        res[i] = OptNN2(d0 = d0, h = h, d1 = d1, m_ep = m_ep)
        logger.info("LogLoss: %10.4f ", res[i])
        np.savez_compressed("res2_NN.npz", res = res)
    """


if False:
    CONFIG['ensemble_list'] = \
    ['btc','btc2','btc3','btc4','svc','svc2','svc3','nn','nn2','nic',
            'mpc','knc','etc','cccv', 'log', 'cetcbag', 'crfcbag']
    X, Xtest = GetDataset('ensemble', ensemble_list = CONFIG['ensemble_list'])
    _, y, _ = LoadData(); del _
    Y = np.array([int(i[-1]) for i in y])
    Y = Y.astype('int32') - 1
    X = X.astype('float32')

if True:
    l_start = .017
    l_stop = 1e-6
    m_start = .9
    m_stop = .999
    max_epochs = 3000
    print l_start, l_stop, m_start, m_stop, max_epochs

    params = dict(
        layers = [
            ('input', layers.InputLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],

        input_shape = (None, 153),
        dropout1_p = .1,
        hidden1_num_units = 500,
        dropout2_p = .2,
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
            EarlyStopping(patience=70),
            ],
        max_epochs = max_epochs,
        verbose = 1,
        )
    clf = NeuralNet(**params)
    clf.fit(X,Y)


