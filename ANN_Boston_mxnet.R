library(mxnet)
library(MASS)
library(caTools)

dt <- setData(Boston, "medv")

Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)

y = as.matrix(Boston.scaled[,14])
y = as.numeric(y)
x = as.numeric(as.matrix(Boston.scaled[,1:13]))
x = matrix(as.numeric(x),ncol=9)

train.x = x
train.y = y
test.x = x
test.y = y

mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=c(5,5), out_node=2, out_activation="softmax", num.round=20, array.batch.size=32, learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy)

preds = predict(model, test.x)
## Auto detect layout of input matrix, use rowmajor..
pred.label = max.col(t(preds))-1
table(pred.label, test.y)