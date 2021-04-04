# install.packages('e1071')
# install.packages('mlbench')
library(e1071)
library(mlbench)

# Read dataset
setwd('C:/Users/a310005/Desktop/DiTella/Data Mining/Clase_1')
getwd()
data <- read.csv('bankruptcy_data_red.csv', sep=';')
data <- na.omit(data)
head(data)
summary(data)
nrow(data)
ncol(data)

# Split dataset in trining and test
test_indexes <- sample(c(1:nrow(data)), 300)
train_data <- data[-test_indexes,]
test_data  <- data[test_indexes,]

# Naive Bayes training model
nb_classifier <- naiveBayes(class ~ ., data = train_data,
                            laplace=0)

# Predictions in test_model 
preds_test <- predict(nb_classifier, newdata = test_data)
table(predicted = preds_test, actual = test_data$class)
print(mean(preds_test == test_data$class))

# Predictions in training_model 
preds_train <- predict(nb_classifier, newdata = train_data)
table(predicted = preds_train, actual = train_data$class)
print(mean(preds_train == train_data$class))

# Plot model accuracy
plot_data <- c(mean(preds_train == train_data$class),
               mean(preds_test == test_data$class))
barplot(plot_data, main='Model Accuracy',
        xlab='Dataset', ylab='Accuracy',
        col='darkred',
        names.arg = c('Train Data', 'Test Data'))

# No son features categoricas, aca no se luce Bayes ingenuo
# Add one smoothing sirve cuando hay pocos datos y algunas conjuntos
# son vacias y nos da probabilidad condicional 0. En este caso
# el dataser tiene columnas no categoricas y creo que se trabaja con 
# media y desviacion, por lo que no mejora nada

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

library(class); library(kernlab)

# Read dataset
setwd('C:/Users/a310005/Desktop/DiTella/Data Mining/Clase_1')
getwd()
data <- read.csv('bankruptcy_data_red.csv', sep=';')
data <- na.omit(data)
head(data)
summary(data)
nrow(data)
ncol(data)

# Removing column to predict
std.data <- data[,-ncol(data)] 
std.data <- na.omit(std.data)
# Conver to numeric every column in dataset
std.data <- sapply( std.data, as.numeric )
std.data <- scale(std.data)

# Split in test and training data
test_indexes <- sample(seq_len(nrow(std.data)), 300) 
train_x <- std.data[-test_indexes,]
train_y <- data[-test_indexes, "class"]
test_x  <- std.data[test_indexes,]
test_y  <- data[test_indexes, "class"]

# Replacing NA for 0
train_x[is.na(train_x)] = 0
train_y[is.na(train_y)] = 0
test_x[is.na(test_x)] = 0
test_y[is.na(test_y)] = 0

# Knn model training
knn_predictions <- knn(train_x, test_x, train_y, k=5)

# Model prediction accuracy analysis in test data
table(preds = knn_predictions, actual = test_y)
print(mean(test_y == knn_predictions))  

# Model prediction accuracy analysis in training data
print(mean(train_y == knn(train_x, train_x, train_y, k=5)))  

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# How Knn performs for different values of k?

library(ggplot2)
library(reshape2)

train_acc <- c()
test_acc  <- c()
k_vals <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            15, 30, 50, 75, 100, 125, 150)

for (k in k_vals) {
    
    print(k)
    
    tmp_tr_pred <- knn(train_x, train_x, train_y, k)
    tmp_ts_pred <- knn(train_x, test_x, train_y, k)
    
    # table(preds = tmp_tr_pred, actual = train_y)
    # table(preds = tmp_tr_pred, actual = test_y)
    print(mean(train_y == tmp_tr_pred))
    print(mean(train_y == tmp_tr_pred))
    train_acc <- c(train_acc, mean(train_y == tmp_tr_pred))
    test_acc  <- c(test_acc, mean(test_y == tmp_ts_pred))
    
}

experiment_data <- data.frame(k = k_vals, train_acc, test_acc)
print(experiment_data)

plot_data <- melt(experiment_data, id.vars="k", value.name="Accuracy")

ggplot(data=plot_data, aes(x=k, y=Accuracy, col=variable)) + geom_line()

max(test_acc)

# k = 9 es el mejor k con test_acc = 0.7833333