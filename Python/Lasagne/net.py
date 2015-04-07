from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
import numpy as np
import theano
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
if False:
    from ml import *
    X, Xtest = GetDataset("text-standardized")
    X = X.astype('float32')
    _, y, _ = LoadData(); del _
    Y = np.array([int(i[-1]) for i in y])
    Y = Y.astype('int32') - 1

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = np.log10(start), np.log10(stop)
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.logspace(self.start, self.stop,
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
params = dict(
    layers = [
        ('input', layers.InputLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape = (None, 93),
    dropout1_p = .1,
    hidden1_num_units = 400,
    dropout2_p = .4,
    hidden2_num_units = 400,
    dropout3_p = .4,
    output_nonlinearity = softmax,
    output_num_units = 9,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(.03)),
    update_momentum = theano.shared(float32(.9)),

    regression = False,
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = .008, stop = .0001),
        AdjustVariable('update_momentum', start = .9, stop = .999),
        ],
    max_epochs = 400,
    verbose = 1,
    )
print "MacOS"
net = NeuralNet(**params)
net.fit(X, Y)
if False:
    kcv = StratifiedKFold(Y, 5, random_state = 314)
    res = np.empty((len(Y), len(np.unique(Y)))); i = 1
    CVScores = []
    for train_idx, valid_idx in kcv:
        logger.info("Running fold %d...", i); i += 1
        net = NeuralNet(**params)
        net.set_params(eval_size = None)
        net.fit(X[train_idx], Y[train_idx])
        res[valid_idx, :] = net.predict_proba(X[valid_idx]) 
        CVScore.append(log_loss(Y[valid_idx], res[valid_idx]))
