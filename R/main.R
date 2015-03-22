library("randomForest")
library("glmnet")
library("parallel")
library("doParallel")
library("foreach")
library("mda")
library("gbm")

setwd("~/Downloads/Otto/")
load("Data.RData")
options(max.print = 100)
nCores = 4
registerDoParallel(nCores)

LogLoss = function(y, yhat, inflate = 0.01)
{
  yhat = (yhat + inflate)/(1 + ncol(yhat)*inflate)
  classes = colnames(yhat)
  prob = numeric(length(y))
  for (i in 1:length(classes)) {
    prob[y == classes[i]] = yhat[,i][y == classes[i]]  
  }
  return(-sum(log(prob))/length(y))
}

Submit = function(yhat, filename)
{
  submission = read.csv("sampleSubmission.csv")
  submission[,-1] = yhat
  write.table(x = submission, file = filename, sep = ",", 
              col.names = TRUE, row.names = FALSE)
}

if (TRUE) {
  rf = randomForest(X[train,], y[train], ntree = 200, 
                    do.trace = TRUE)
  yhat = predict(rf, X[valid,], type = "prob")
  yhat = predict(rf, Xtest, type = "prob")
  yhat = (yhat + 0.001)/1.009
}

if (FALSE) {
  lg = cv.glmnet(x = X[train,], y = y[train], family = "multinomial", 
                 type.measure = "deviance", nfolds = 5, standardize = FALSE)
  yhat = predict(lg, X[valid,], type = "response")
  yhat = yhat[,,1]  
}

if (TRUE) {
  gb = gbm.fit(x = X[train, ], y = y[train], distribution = "multinomial",
               shrinkage = 0.001, n.trees = 50)
  yhat = predict(gb, X[valid,], n.trees = 50, type = "response")
  yhat = yhat[,,1]
}

if (TRUE) {
  ma = mars(x = X[train,], y = y[train])
}