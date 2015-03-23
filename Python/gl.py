import graphlab as gl
import numpy as np
from sklearn.metrics          import log_loss
from sklearn.cross_validation import StratifiedKFold
from utils import *
import time
import os
import math
import random

def TransformLabel(y):
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

    sample = gl.SFrame.read_csv('../Data/sampleSubmission.csv')
    assert sample.num_rows() == preds.num_rows()

    yhat = preds.to_dataframe()
    yhat.to_csv(file_name, index = None)

def gridLogLoss(y, yhat):
    return min([log_loss(y, yhat, eps = x) 
	        for x in .0001*2.**np.arange(-10,10)])

# Load the data
def FinalModel():
    X = gl.SFrame.read_csv('../Data/train.csv')
    Xtest = gl.SFrame.read_csv('../Data/test.csv')

    del X['id']
    model = gl.boosted_trees_classifier.create(X, target = 'target',
					  max_iterations = 200,
					  row_subsample = 0.9,
                                          step_size = 0.25,
                                          validation_set = None)

    makeSubmission("GBM_200iter_Subsample0.9_stepsize0.25", Xtest, model)

def LogLossGL(model, train, test):
    m = model(train, target = 'target', validation_set = None)

    preds = m.predict_topk(test, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', 
                                                                '')
    preds = preds.sort('id')
    del preds['id']
    yhat = preds.to_dataframe().as_matrix()
    y = np.array(test['target'])
    y = TransformLabel(y)
    res = logLossAdjGrid(y, yhat)
    return {"Log Loss" : res}

def MulticlassLogLoss(model, test):
    preds = model.predict_topk(test, output_type='probability', k=9)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    preds['target'] = test['target']
    neg_log_loss = 0
    for row in preds:
        label = row['target']
        neg_log_loss += - math.log(row[label])
    return  neg_log_loss / preds.num_rows()

def MulticlassLogLoss2(model, test):
    preds = model.predict_topk(test, output_type='probability', k=9)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    del preds['id']
    preds = preds.to_dataframe().as_matrix()
    y = np.array(test['target'])
    y = TransformLabel(y)
    return logLossAdjGrid(y, preds) 


def EvaluateLogLoss(model, train, valid):
    return {'train_logloss' : MulticlassLogLoss(model, train),
            'valid_logloss' : MulticlassLogLoss(model, valid)}

def GetKFold(y):
    n = len(y)
    skf = StratifiedKFold(y, n_folds = 5)
    for train_index, valid_index in skf:
        train_bool = np.array(np.zeros(n), dtype = bool)
        train_bool[train_index] = True
        valid_bool = np.array(np.zeros(n), dtype = bool)
        valid_bool[valid_index] = True
        yield gl.SArray(train_bool), gl.SArray(valid_bool)

def AnalyzeCV(res, file_name, cols):
    res_all = res[0]['summary']
    training_colnames = ['train_logloss']
    validation_colnames = ['valid_logloss']
    for i in range(1,5):
        res_all = res_all.join(res[i]['summary'], on = cols, how = 'inner')
        training_colnames.append('training_accuracy.' + str(i))
        validation_colnames.append('validation_accuracy.' + str(i))
    training_acc = res_all.select_columns(training_colnames).to_dataframe()
    training_acc_mean = np.mean(training_acc.as_matrix(), axis = 1)
    validation_acc = res_all.select_columns(validation_colnames).to_dataframe()
    validation_acc_mean = np.mean(validation_acc.as_matrix(), axis = 1)
    res_all[int(np.argmax(training_acc_mean))]
    res_all.save(filename = file_name)
    return res_all[int(np.argmax(validation_acc_mean))]
    

if True:
    #max_iter = int(os.environ['max_iter'])
    cols = ['model_id']
    row_subsample = [0.8,1]
    step_size = [0.8, 1.0]
    if type(row_subsample) is list: cols.append('row_subsample')
    if type(step_size)     is list: cols.append('step_size')
    write("Lon Running Model for Max Iter: " + str(max_iter) + "\n")
    X = gl.SFrame.read_csv('../Data/train.csv'); del X['id']
    y = np.array(X['target']); res = []; fold = 1
    
    for train_bool, valid_bool in GetKFold(y):
        time_before = time.time()
        job = gl.model_parameter_search(gl.boosted_trees_classifier.create, 
	        training_set = X[train_bool], target = 'target', 
            validation_set = X[valid_bool], evaluator = EvaluateLogLoss, 
	    row_subsample = row_subsample,
            max_iterations = max_iter,
            step_size = step_size,
            )
        job_result = job.get_results()
        print job_result
        write("Fold " + str(fold) + " Running Time: " + 
                str(time.time() - time_before) + "\n")
        fold += 1
        res.append(job_result)
    print AnalyzeCV(res, "GBM_CV1"+str(max_iter) + ".csv", 
            cols)

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
