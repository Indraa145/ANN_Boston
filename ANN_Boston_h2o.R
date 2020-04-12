library(h2o)
library(MASS)
library(caTools)

localH2O = h2o.init(ip="127.0.0.1", port = 50001, 
                    startH2O = TRUE, nthreads=-1)

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
train <- Boston.scaled[Boston.split, ]
test <- Boston.scaled[!Boston.split, ]

y = names(train)[14]
x = names(train)[1:13]

train[,y] = as.factor(train[,y])
test[,y] = as.factor(train[,y])

model = h2o.deeplearning(x=x, 
                         y=y, 
                         training_frame=train, 
                         validation_frame=test, 
                         distribution = "multinomial",
                         activation = "RectifierWithDropout",
                         hidden = c(10,10,10,10),
                         input_dropout_ratio = 0.2,
                         l1 = 1e-5,
                         epochs = 50)

print(model)