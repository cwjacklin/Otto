import numpy as np
import pandas as pd
from scipy.stats import uniform, randint

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(1337) # for reproducibility

from data import *

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
                y = np_utils.to_categorical(y)
    return y, encoder
X, Xtest = GetDataset('original')
_, y, _ = LoadData(); del _
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = np_utils.to_categorical(y)

nb_classes = y.shape[1]
dims = X.shape[1]

def OptKeras(h1, h2, h3, d1, d2, d3, d4, ne):
    model = Sequential()
    model.add(Dense(dims, h1, init='glorot_uniform'))
    model.add(PReLU((h1,)))
    model.add(BatchNormalization((h1,)))
    model.add(Dropout(d1))

    model.add(Dense(h1, h2, init='glorot_uniform'))
    model.add(PReLU((h2,)))
    model.add(BatchNormalization((h2,)))
    model.add(Dropout(d2))

    model.add(Dense(h2, h3, init='glorot_uniform'))
    model.add(PReLU((h3,)))
    model.add(BatchNormalization((h3,)))
    model.add(Dropout(d3))

    model.add(Dense(h3, nb_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print("Training model...kkk")

    model.fit(X, y, nb_epoch = ne, batch_size=1024, validation_split=0.2)

OptKeras(256, 512, 1024, .5, .5, .5, .5, 400)
if __name__ == '_main__':
    for i in xrange(1000):
        print "Training model", i
        d1 = uniform(.1, .6).rvs() 
        d2 = uniform(.1, .6).rvs()
        d3 = uniform(.1, .6).rvs()
        h1 = randint(200, 2000).rvs()
        h2 = randint(200, 2000).rvs()
        h3 = randint(200, 2000).rvs()
        ne = randint(50, 1000).rvs()
        OptKeras(h1, h2, h3, d1, d2, d3, ne)
