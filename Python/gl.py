import graphlab as gl
from sklearn.metrics import log_loss
import numpy as np

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
X = gl.SFrame.read_csv('../data/train.csv')
Xtest = gl.SFrame.read_csv('../data/test.csv')
sample = gl.SFrame.read_csv('../data/sampleSubmission.csv')

del X['id']
#Xtrain, Xvalid = X.random_split(.7, seed = 1)
#del X
#y = transformLabel(array(Xvalid['target']))

# Train a model
#row_subsample = 1.; print row_subsample
#max_iteration = 1000; print max_iteration
model = gl.boosted_trees_classifier.create(X, target = 'target',
		                      max_iterations = 3000,
				      row_subsample = 0.8,
				      step_size = 0.1)

"""
makeSubmission("GBM_200iter", Xtest, model)

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
