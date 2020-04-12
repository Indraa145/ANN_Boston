library(neuralnet)
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

# Train-test split
Boston.split <- sample.split(Boston$medv, SplitRatio = 0.75)
Boston.train.scaled <- Boston.scaled[Boston.split, ]
Boston.test.scaled <- Boston.scaled[!Boston.split, ]

# neuralnet doesn't accept resp~. (dot) notation
# so a utility function to create a verbose formula is used
Boston.nn.fmla <- generate.full.fmla("medv", Boston)

# 2 models, one with 2 layers of 5 and 3
# the second with one layer of 8
# linear output is used for a regression problem
Boston.nn.5.3 <- neuralnet(Boston.nn.fmla
                           , data=Boston.train.scaled
                           , hidden=c(5,3)
                           , linear.output=TRUE)

Boston.nn.8 <- neuralnet(Boston.nn.fmla
                         , data=Boston.train.scaled
                         , hidden=8
                         , linear.output=TRUE)

# Predict - remember the output is scaled
Boston.5.3.preds.scaled <- neuralnet::compute(Boston.nn.5.3
                                              , Boston.test.scaled[,1:13])
Boston.8.preds.scaled <- neuralnet::compute(Boston.nn.8
                                            , Boston.test.scaled[,1:13])

Boston.5.3.preds <- Boston.5.3.preds.scaled$net.result
Boston.8.preds <- Boston.8.preds.scaled$net.result

cor(Boston.5.3.preds, Boston.test.scaled$medv)
cor(Boston.8.preds, Boston.test.scaled$medv)