import graphlab as gl
import numpy as np
from sklearn.metrics          import log_loss
from sklearn.cross_validation import StratifiedKFold
from utils import *
import time

def transformLabel(y):
    res = np.empty(shape = (len(y), 9))
    for i in range(9):
	res[:,i] = (y == 'Class_' + str(i+1)) + 0
    return res
# Make submission
def makeSubmission(file_name, Xtest, model):
    preds = model.predict_topk(Xtest, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds = preds.sort('id')

    assert sample.num_rows() == preds.num_rows()

    yhat = preds.to_dataframe()
    yhat.to_csv(file_name, index = None)

def gridLogLoss(y, yhat):
    return min([log_loss(y, yhat, eps = x) 
	        for x in .0001*2.**np.arange(-10,10)])

# Load the data
def FinalModel(X, Xtest):
    X = gl.SFrame.read_csv('../data/train.csv')
    Xtest = gl.SFrame.read_csv('../data/test.csv')
    sample = gl.SFrame.read_csv('../data/sampleSubmission.csv')

    del X['id']
    model = gl.boosted_trees_classifier.create(X, target = 'target',
					  max_iterations = 200,
					  row_subsample = 0.8)

    makeSubmission("GBM_200iter_Subsample.8", Xtest, model)

def GetKFold(y):
    n = len(y)
    skf = StratifiedKFold(y, n_folds = 5)
    for train_index, valid_index in skf:
        train_bool = np.array(np.zeros(n), dtype = bool)
        train_bool[train_index] = True
        valid_bool = np.array(np.zeros(n), dtype = bool)
        valid_bool[valid_index] = True
        yield gl.SArray(train_bool), gl.SArray(valid_bool)

if True:
    X = gl.SFrame.read_csv('../data/train.csv')
    del X['id']
    y = np.array(X['target'])
    res = []
    for train_bool, valid_bool in GetKFold(y):
        time_before = time.time()
        job = gl.model_parameter_search(gl.boosted_trees_classifier.create, 
	        training_set = X[train_bool], target = 'target', 
            validation_set = X[valid_bool],
	        row_subsample = [0.8, 0.9, 1],
            max_iterations = 200,
            step_size = [0.25, 0.5, 1.]
            )
        job_result = job.get_results()
        print job_result
        write("Fold 1 Running Time: " + str(time.time() - time_before) + "\n")
        res.append(job_result)

"""
yh = model.predict(Xvalid)
print gl.evaluation.accuracy(Xvalid['target'], yh)
yhat = model.predict_topk(Xvalid, output_type = 'probability', k = 9)
yhat['id'] = yhat['id'].astype(int)
yhat = yhat.unstack(['class', 'probability'], 'probs').unpack('probs')
yhat = yhat.sort('id')
del yhat['id']
yhat = yhat.to_dataframe().as_matrix()
print gridLogLoss(y, yhat)

dl = gl.deeplearning.create(Xtrain, target = 'target', 
	                    network_type = 'perceptrons')
dl.layers[0].num_hidden_units = 40
nn = gl.neuralnet_classifier.create(Xtrain, target = 'target', 
	max_iterations = 200, network = dl, l2_regularization = 0.005,
	momentum = 0.5)
"""
