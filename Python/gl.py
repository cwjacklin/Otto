import graphlab as gl
import numpy as np
from sklearn.metrics          import log_loss
from sklearn.cross_validation import StratifiedKFold
from utils import *
import time
import os
import math
import random
import graphlab.aggregate as agg
import argparse

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

if True:
    data = np.load("../Data/Data.npz")
    train = data['train']
    valid = data['valid']
    X = data['X']
    X = gl.SFrame(pd.DataFrame(X))
    X['target'] = data['y']
    n = len(X)
    train_bool = np.array(np.zeros(n), dtype = bool)
    train_bool[train] = True
    valid_bool = np.array(np.zeros(n), dtype = bool)
    valid_bool[valid] = True
    train = gl.SArray(train_bool)
    valid = gl.SArray(valid_bool)
    
    model = gl.boosted_trees_classifier.create(
                X[train], target = 'target', 
                max_iterations     = 250,
                max_depth          = 10,
                min_child_weight   = 4,
                row_subsample      = .9,
                min_loss_reduction = 1,
                column_subsample = .8,
                validation_set = None)
    makeSubmission("yhat_gbm2.csv", X[valid], model)


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

def StackCVRes(res):
    cv_stack = res[0]['summary']
    for i in xrange(1, len(res)):
        cv_stack = cv_stack.join(res[i]['summary'], how = 'outer')
    return cv_stack

def AnalyzeCV(res, col_name, n_cv_vars):
    cv_res = StackCVRes(res)
    key_columns = cv_res.column_names()[:(1 + n_cv_vars)]
    cv = cv_res.groupby(key_columns, 
            operations = {'mean_' + col_name : agg.MEAN(col_name), 
                          'std_'  + col_name : agg.STD(col_name)})
    cv = cv.sort('mean_' + col_name)
    cv.print_rows(len(cv))
    return cv

def main(config):
    max_iter = config.maxiter
    row_subsample = [.8, 1.]
    column_subsample = [.5, 1.]
    step_size = [.1, .2, .5, 1.]
    max_depth = [6,8]
    
    Write("Running Model for Max Iter: " + str(max_iter) + "\n")
    X = gl.SFrame.read_csv('../Data/train.csv'); del X['id']
    y = np.array(X['target']); res = []; fold = 1
    
    for train_bool, valid_bool in GetKFold(y):
        time_before = time.time()
        job = gl.model_parameter_search(gl.boosted_trees_classifier.create, 
	    training_set     = X[train_bool], target = 'target', 
            validation_set   = X[valid_bool], evaluator = EvaluateLogLoss, 
	    row_subsample    = row_subsample,
            column_subsample = column_subsample,
            max_depth        = max_depth,
            max_iterations   = max_iter,
            step_size        = step_size
            )
        job_result = job.get_results()
        print job_result
        Write("Fold " + str(fold) + " Running Time: " + 
                str(time.time() - time_before) + "\n")
        fold += 1
        res.append(job_result)
    cv =  AnalyzeCV(res, "valid_logloss",4)
    cv.save("CV_GL_" + str(max_iter) + ".csv")

if __name__ == '_main__':
   parser = argparse.ArgumentParser(description = "Parameters for the script.")
   parser.add_argument('-m', "--maxiter", 
                        help = "Max Iteration Parameter", type = int) 
   config = parser.parse_args()
   max_iter =  config.maxiter
   #main(config)
