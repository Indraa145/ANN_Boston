# Deep Learning - ANN on Boston Dataset
## Deep Learning Homework 3 No. 8 | Indra Imanuel Gunawan - 20195118
This is R implementation on the Boston Dataset. There are 5 R packages that is used in this experiments, which are:
1. Rneuralnet
2. h20
3. mxnet
4. TensorFlow & KerasR
5. TensorFlow & Keras
I will explain each of the code in this report.

## Rneuralnet
First, load the neuralnet libary and other necessary libraries
```R
library(neuralnet)
library(MASS)
library(caTools)
```
Load the data, and normalize it
```R
dt <- setData(Boston, "medv")

Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)
```
Split it into training set and test set
```R
# Train-test split
Boston.split <- sample.split(Boston$medv, SplitRatio = 0.75)
Boston.train.scaled <- Boston.scaled[Boston.split, ]
Boston.test.scaled <- Boston.scaled[!Boston.split, ]
```

Build the neural network models (There are 2 models to test here, with differenct architecture)
```R
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
```
Build the predictor and see its performance
```R
Boston.5.3.preds.scaled <- neuralnet::compute(Boston.nn.5.3
                                              , Boston.test.scaled[,1:13])
Boston.8.preds.scaled <- neuralnet::compute(Boston.nn.8
                                            , Boston.test.scaled[,1:13])

Boston.5.3.preds <- Boston.5.3.preds.scaled$net.result
Boston.8.preds <- Boston.8.preds.scaled$net.result

cor(Boston.5.3.preds, Boston.test.scaled$medv)
cor(Boston.8.preds, Boston.test.scaled$medv)
```

## H2O
Load the h2o library and the other necessary libraries as well
```R
library(h2o)
library(MASS)
library(caTools)
```
Initialize the h2o
```R
localH2O = h2o.init(ip="127.0.0.1", port = 50001, 
                    startH2O = TRUE, nthreads=-1)
```
Load and normalize the data
```R
dt <- setData(Boston, "medv")

Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)
```
Split it into training and test dataset, also set the x and y
```R
# Train-test split
Boston.split <- sample.split(Boston$medv, SplitRatio = 0.75)
train <- Boston.scaled[Boston.split, ]
test <- Boston.scaled[!Boston.split, ]

y = names(train)[14]
x = names(train)[1:13]

train[,y] = as.factor(train[,y])
test[,y] = as.factor(train[,y])
```

Build and run the model
```R
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
```

## MXNET
Load the mxnet library and the other necessary libraries
```R
library(mxnet)
library(MASS)
library(caTools)
```
Load the data and normalize it
```R
dt <- setData(Boston, "medv")

Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)
```
Set the X and Y, and set the training and test dataset
```R
y = as.matrix(Boston.scaled[,14])
y = as.numeric(y)
x = as.numeric(as.matrix(Boston.scaled[,1:13]))
x = matrix(as.numeric(x),ncol=14)

train.x = x
train.y = y
test.x = x
test.y = y
```
Build and run the model, and the predictor as well, to see the model performance
```R
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=c(5,5), out_node=2, out_activation="softmax", num.round=20, array.batch.size=32, learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy)

preds = predict(model, test.x)
## Auto detect layout of input matrix, use rowmajor..
pred.label = max.col(t(preds))-1
table(pred.label, test.y)
```

## TensorFlow & KerasR
Load TensorFlow and KerasR library along with other necessary libraries as well
```R
library(tensorflow)
library(kerasR)
library(MASS)
library(caTools)
```
Load the data and normalize it
```R
dt <- setData(Boston, "medv")

Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)
```
Split the data into training and test dataset, also set the x and y
```R
# Train-test split
Boston.split <- sample.split(Boston$medv, SplitRatio = 0.75)
tf_train <- Boston.scaled[Boston.split, ]
tf_test <- Boston.scaled[!Boston.split, ]

X_train = as.matrix(tf_train[,1:13])
X_test = as.matrix(tf_test[,1:13])
y_train = as.matrix(tf_train[,14])
y_test = as.matrix(tf_test[,14])
```
Build the neural network
```R
n_units = 512 

mod <- Sequential()
mod$add(Dense(units = n_units, input_shape = dim(X_train)[2]))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(2))
mod$add(Activation("softmax"))
```
Compile the model and fit the data into the model
```R
keras_compile(mod, loss = 'categorical_crossentropy', optimizer = RMSprop())

keras_fit(mod, X_train, Y_train, batch_size = 32, epochs = 15, verbose = 2, validation_split = 1.0)

#Validation
Y_test_hat <- keras_predict_classes(mod, X_test)
table(y_test, Y_test_hat)
print(c("Mean validation accuracy = ",mean(y_test == Y_test_hat)))
```

## TensorFlow and Keras
Load all of the necessary libraries
```R
library(magrittr)
library(tensorflow)
library(keras)
library(MASS)
library(caTools)
```
Load and normalize the dataset
```R
dt <- setData(Boston, "medv")

Boston.scaled <- as.data.frame(scale(Boston))
min.medv <- min(Boston$medv)
max.medv <- max(Boston$medv)
# response var must be scaled to [0 < resp < 1]
Boston.scaled$medv <- scale(Boston$medv
                            , center = min.medv
                            , scale = max.medv - min.medv)
```
Split the data into training and test dataset, also set the x and y
```R
# Train-test split
Boston.split <- sample.split(Boston$medv, SplitRatio = 0.75)

#training set
tf_train <- Boston.scaled[Boston.split, ]

#test set
tf_test <- Boston.scaled[!Boston.split, ]

X_train = as.matrix(tf_train[,1:13])
X_test = as.matrix(tf_test[,1:13])
y_train = as.matrix(tf_train[,14])
y_test = as.matrix(tf_test[,14])
```
Build the neural network
```R
model <- keras_model_sequential() 

n_units = 100
model %>% 
  layer_dense(units = n_units, 
              activation = 'relu', 
              input_shape = dim(X_train)[2]) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')
```
Compile the model and fit the data into the model
```R
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

model %>% fit(
  X_train, Y_train, 
  epochs = 5, batch_size = 32, verbose = 1, 
  validation_split = 0.1
)
```
