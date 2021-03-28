#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 1 ##
install.packages('mlbench')
# library(e1071)
# library(mlbench)

data(HouseVotes84, package = "mlbench")

# Separo un conjunto de entrenamiento y otro no usado para entrenar
left_out_indexes <- sample(c(1:nrow(HouseVotes84)), 50)
train_data <- HouseVotes84[-left_out_indexes,]
left_out_data  <- HouseVotes84[left_out_indexes,]

# Entreno un modelo de bayes ingenuo sobre train
nb_classifier <- naiveBayes(Class ~ ., data = train_data)

# Veo cómo fue sobre el conjunto de entrenamiento y sobre el no usado para entrenar
preds <- predict(nb_classifier, newdata = left_out_data)
table(predicted = preds, actual = left_out_data$Class)
print(mean(preds == left_out_data$Class))

# Cómo lo hizo sobre los datos con los que se entrenó el modelo
print(mean(predict(nb_classifier, newdata = train_data) == train_data$Class))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 2 ##

# Probemos cómo funciona knn
# install.packages('ISLR')
# install.packages('kernlab')
# install.packages('class')
library(class); library(ISLR); library(kernlab)

data(spam)
std.spam <- scale(spam[,-ncol(spam)]) # Quitamos la columna a predecir
summary(std.spam)
apply(std.spam, 2, sd)

# Separo un conjunto de entrenamiento y otro no usado para entrenar
left_out_indexes <- sample(seq_len(nrow(std.spam)), 920) # %20 para testear. 
train_X <- std.spam[-left_out_indexes,]
left_out_X  <- std.spam[left_out_indexes,]
train_y <- spam[-left_out_indexes, "type"]
left_out_y  <- spam[left_out_indexes, "type"]

# Entreno un modelo de vecinos más cercanos
knn_predictions <- knn(train_X, left_out_X, train_y, k=5)

# Veo cómo fue sobre el conjunto de entrenamiento y sobre el no usado para entrenar
table(preds = knn_predictions, actual = left_out_y)
print(mean(left_out_y == knn_predictions))  # Performance en el conjunto de datos separado

print(mean(train_y == knn(train_X, train_X, train_y, k=5)))  # Performance en el conjunto de datos "conocido"

# Cómo le hubiera ido a Naïve Bayes?
nb_classifier <- naiveBayes(type ~ ., data=spam[-left_out_indexes,])
print(mean((left_out_y == predict(nb_classifier, newdata=spam[left_out_indexes,])))) # ¿Por qué funcionará peor?

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 3 ##

## Probemos cómo anda con distintos valores de k

library(ggplot2)
library(reshape2)

train_acc <- c()
left_out_acc  <- c()
k_vals <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            15, 30, 50, 75, 100, 125, 150)

for (k in k_vals) {

  print(k)

  tmp_tr_pred <- knn(train_X, train_X, train_y, k)
  tmp_ts_pred <- knn(train_X, left_out_X, train_y, k)

  train_acc <- c(train_acc, mean(train_y == tmp_tr_pred))
  left_out_acc  <- c(left_out_acc, mean(left_out_y == tmp_ts_pred))

}

experiment_data <- data.frame(k = k_vals, train_acc, left_out_acc)
print(experiment_data)

plot_data <- melt(experiment_data, id.vars="k", value.name="Accuracy")

ggplot(data=plot_data, aes(x=k, y=Accuracy, col=variable)) + geom_line()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
