library(tensorflow)
library(kerasR)
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
tf_train <- Boston.scaled[Boston.split, ]
tf_test <- Boston.scaled[!Boston.split, ]

X_train = as.matrix(tf_train[,1:13])
X_test = as.matrix(tf_test[,1:13])
y_train = as.matrix(tf_train[,14])
y_test = as.matrix(tf_test[,14])

#Y_train <- to_categorical(y_train,2)

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

keras_compile(mod, loss = 'categorical_crossentropy', optimizer = RMSprop())

keras_fit(mod, X_train, Y_train, batch_size = 32, epochs = 15, verbose = 2, validation_split = 1.0)

#Validation
Y_test_hat <- keras_predict_classes(mod, X_test)
table(y_test, Y_test_hat)
print(c("Mean validation accuracy = ",mean(y_test == Y_test_hat)))