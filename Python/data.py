import pandas as pd
import numpy as np

np.random.seed(0)
train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')

index = np.arange(len(train))
np.random.shuffle(index)
train = train.loc[index]
train.index = np.arange(len(train))

y = train.target.values
X = train.drop(['id', 'target'], axis = 1).as_matrix()
Xtest = test.drop('id', axis = 1).as_matrix()

cut = len(y)*7/10
train = np.arange(cut)
valid = np.arange(cut, len(y))
np.savez_compressed(file = "../Data/Data.npz", X = X, Xtest = Xtest, y = y,
                    train = train, valid = valid)
