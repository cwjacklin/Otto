import sys
sys.path.insert(0, '../Library/MLP/')
from autoencoder            import *
from multilayer_perceptron  import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def pseudoAccuracy(yint, yhat):
    P = confusion_matrix(yint, yhat)
    n = np.sum(P)
    diag = 0.
    for i in xrange(len(P)):
        diag += np.max(P)
        max_index = np.argwhere(P == np.max(P))
        i = max_index[0][0]; j = max_index[0][1];
        P[i,:] = 0; P[:,j] = 0
        # print P, "\n"
    return diag/n

if True:
    n_init = 80;    print "n_init: ", n_init, "\n"
    max_iter = 500; print "max_iter: ", max_iter, "\n"
    time_before = time.time()
    km = KMeans(n_clusters = 9, init = 'k-means++', n_init = n_init, 
            max_iter = max_iter, precompute_distances = False, verbose = 0,
            n_jobs = 12)
    write("Not Using Xtest: \n")
    km.fit(X)
    yhat = km.predict(X) + 1
    yint = np.array([int(i[-1]) for i in y])
    write("Running Time: " + str(time.time() - time_before) + "\n")
    write(str(pseudoAccuracy(yint, yhat)))

if False:
    ae = Autoencoder(n_hidden = 100,             algorithm = 'l-bfgs', 
                learning_rate = 'invscaling', sparsity_param = 0.1, 
                batch_size    = 500,              max_iter = 200,
                      verbose = True)
    #ae.fit(X)
    #Xae = ae.transform(X)

    ae = MultilayerPerceptronAutoencoder(
             hidden_layer_sizes = (100,),   
                      algorithm = 'l-bfgs',       beta = 3, 
                 sparsity_param = 0.5,      batch_size = 200,
                  learning_rate = 'invscaling', max_iter = 200,
                        shuffle = True,        verbose = True,
                     warm_start = True)
    #Xae = ae.fit_transform(X)
    clf = SGDClassifier()
    clf.fit(X,y)
    print clf.score(X,y)
    clf.fit(Xae, y)
    print clf.score(Xae, y)


    mlp = MultilayerPerceptronClassifier(
            hidden_layer_sizes = 100, max_iter = 200, alpha = 0.00001,
            verbose = True)
    mlp.fit(X[train], y[train])
    mlp.score(X[valid], y[valid])
