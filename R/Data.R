library(randomForest)

setwd("~/Downloads/Otto/")
data = read.csv("train.csv")
Xtest = read.csv("test.csv")
data$id = NULL
Xtest$id = NULL
Xtest = as.matrix(Xtest)

y = data$target
data$target = NULL
X = as.matrix(data)
options(max.print = 1000)
rm(data)
set.seed(1)
index = sample(length(y), length(y))
train = index[1:(0.7*length(index))]
valid  = index[(0.7*length(index)):length(index)]

save(file = "Data.RData", X, y, train, valid, Xtest)
