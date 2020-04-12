library(magrittr)
library(tensorflow)
library(keras)
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

#training set
tf_train <- Boston.scaled[Boston.split, ]

#test set
tf_test <- Boston.scaled[!Boston.split, ]

X_train = as.matrix(tf_train[,1:13])
X_test = as.matrix(tf_test[,1:13])
y_train = as.matrix(tf_train[,14])
y_test = as.matrix(tf_test[,14])

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