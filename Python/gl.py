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
from sklearn.base import BaseEstimator

class BoostedTreesClassifier(BaseEstimator):
    """
    Scikit-learn wrapper of Graph Lab Boosted Trees Classifier
    
    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations for boosting. Each iteration results 
        in the creation of an extra tree.
    
    max_depth : float, optional
        Maximum depth of a tree.

    class_weights : {dict, auto}, optional
        Weights the examples in the training data according to the given class
        weights. If set to None, all classes are supposed to have weight one.
        The auto mode set the class weight to be inversely proportional to
        number of examples in the training data with the given class.

    step_size : float, [0,1], optional
        Step size (shrinkage) used in update to prevents overfitting. It shrinks
        the prediction of each weak learner to make the boosting process more
        conservative. The smaller the step size, the more conservative the
        algorithm will be. Smaller step_size work well when max_iterations is
        large.

    min_loss_reduction : float, optional
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree. The larger it is, the more conservative the algorithm
        will be.

    min_child_weight : float, optional
        This controls the minimum number of instances needed for each leaf. The
        larger it is, the more conservative the algorithm will be. Set it larger
        when you want to prevent overfitting. Formally, this is minimum sum of
        instance weight (hessian) in each leaf. If the tree partition step
        results in a leaf node with the sum of instance weight less than
        min_child_weight, then the building process will give up further
        partitioning. For a regression task, this simply corresponds to minimum
        number of instances needed to be in each node.

    row_subsample : float, optional
        Subsample the ratio of the training set in each iteration of tree
        construction. This is called the bagging trick and can usually help
        prevent overfitting. Setting this to a value of 0.5 results in the model
        randomly sampling half of the examples (rows) to grow each tree.

    column_subsample : float, optional
        Subsample ratio of the columns in each iteration of tree construction.
        Like row_subsample, this can also help prevent model overfitting.
        Setting this to a value of 0.5 results in the model randomly sampling
        half of the columns to grow each tree.

    validation_set : SFrame, optional
        A dataset for monitoring the model's generalization performance. For
        each row of the progress table, the chosen metrics are computed for both
        the provided training dataset and the validation_set. The format of this
        SFrame must be the same as the training set. By default this argument is
        set to 'auto' and a validation set is automatically sampled and used for
        progress printing. If validation_set is set to None, then no additional
        metrics are computed. This is computed once per full iteration. Large
        differences in model accuracy between the training data and validation
        data is indicative of overfitting. The default value is 'auto'.

    verbose : boolean, optional
        Print progress information during training (if set to true).

    """
    
    def __init__(self, min_loss_reduction = 0,      class_weights = None, 
                       step_size          = 0.3, min_child_weight = 0.1, 
                       column_subsample   = 1,      row_subsample = 1,
                       max_depth          = 6,     max_iterations = 10,
                       verbose            = True,      validation = None):
        self.min_loss_reduction = min_loss_reduction
        self.class_weights      = None
        self.step_size          = step_size
        self.min_child_weight   = min_child_weight
        self.column_subsample   = column_subsample
        self.row_subsample      = row_subsample
        self.max_depth          = max_depth
        self.max_iterations     = max_iterations
        self.verbose            = verbose
        self.validation         = validation

    def fit(self, X, y):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        X: {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features

        y: array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            return self.
        """
        if type(self.validation) is tuple:
            Xvalid, yvalid = self.validation
            Xvalid = gl.SFrame(pd.DataFrame(Xvalid))
            Xvalid['target'] = yvalid
            self.validation = Xvalid
       
        X = gl.SFrame(pd.DataFrame(X))
        X['target'] = y
        self.model = gl.boosted_trees_classifier.create(
                X, target = 'target', 
                validation_set      = self.validation,
                min_loss_reduction  = self.min_loss_reduction,
                class_weights       = self.class_weights,
                step_size           = self.step_size,
                min_child_weight    = self.min_child_weight,
                column_subsample    = self.column_subsample,
                row_subsample       = self.row_subsample,
                max_depth           = self.max_depth,
                max_iterations      = self.max_iterations,
                verbose             = self.verbose)
        self.num_class = len(np.unique(y))
    
    def predict_proba(self, X):
        """
        Probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model
        """

        X = gl.SFrame(pd.DataFrame(X))
        preds = self.model.predict_topk(X, output_type = 'probability', 
                                   k = self.num_class)
        preds['id'] = preds['id'].astype(int) + 1
        preds = preds.unstack(['class', 'probability'], 'probs').unpack(
                                'probs', '')
        preds = preds.sort('id')
        del preds['id']
        return preds.to_dataframe().as_matrix()
    
    def predict(self, X):
        """
        Class estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        yhat: array-like, shape = (n_samples, )
            Returns the predicted class of the sample
        """
        X = gl.SFrame(pd.DataFrame(X))
        yhat = self.model.predict(X)
        return np.array(yhat)

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

def GetYValid():
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
                max_iterations     = 4,
                max_depth          = 10,
                min_child_weight   = 4,
                row_subsample      = .9,
                min_loss_reduction = 1,
                column_subsample = .8,
                validation_set = None)
    #makeSubmission("yhat_gbm2.csv", X[valid], model)


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
    step_size = [.02, .05, .1, .2, .5, 1.]
    max_depth = [6, 8, 10, 12]
    
    Write("Running Model for Max Iter: " + str(max_iter) + "\n")
    X = gl.SFrame.read_csv('../Data/train.csv'); del X['id']
    y = np.array(X['target']); res = []; fold = 1
    
    for train_bool, valid_bool in GetKFold(y):
        time_before = time.time()
        job = gl.model_parameter_search(gl.boosted_trees_classifier.create, 
	    training_set     = X[train_bool], target = 'target', 
            validation_set   = X[valid_bool], evaluator = EvaluateLogLoss, 
	    row_subsample    = row_subsample,
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
    cv =  AnalyzeCV(res, "valid_logloss", 3)
    cv.save("../Jobs/GL/CV_GL_" + str(max_iter) + ".csv")


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description = "Parameters for the script.")
   parser.add_argument('-m', "--maxiter", 
                        help = "Max Iteration Parameter", type = int) 
   config = parser.parse_args()
   max_iter =  config.maxiter
   main(config)
