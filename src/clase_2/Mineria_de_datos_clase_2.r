rm(list=ls())

setwd(...)  # Deben definir su directorio de trabajo
library(class)
library(dplyr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 1 ##

# Holdout set

data_set <- read.table("bankruptcy_data_red.txt", header=TRUE, sep="\t")
data_set$class <- factor(ifelse(data_set$class, "bankruptcy", "no_bankruptcy"))

# Separo en training, validation y testing
holdout_index  <- sample(c(1:nrow(data_set)), 200)  # Aproximadamente un 20% de las obs.
val_data    <- data_set[holdout_index,]
train_data  <- data_set[-holdout_index,]

# Entreno con distintos valores K y valido
k_vals <- c(1, seq(5, 100, by=5))
vld_accuracy <- data.frame()

for (k in k_vals) {
    print(k)
    tmp_vd_pred <- knn(train_data[,-ncol(train_data)],
                       val_data[,-ncol(val_data)], train_data$class, k)

    vld_accuracy  <- rbind(vld_accuracy,
                           data.frame(k = k, acc = mean(val_data$class == tmp_vd_pred)))
}

# Veo la mejor configuración
vld_accuracy[which.max(vld_accuracy$acc),]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 2 ##

# LOOCV

vld_accuracy <- data.frame()

for (k in k_vals) {

    print(k)
    
    for (i in c(1:nrow(data_set))) {  # Ahora itero sobre cada observación

        train_data <- data_set[-i,]
        val_data  <- data_set[i,]

        tmp_vd_pred <- knn(train_data[,-ncol(train_data)],
                           val_data[,-ncol(val_data)], train_data$class, k)

        vld_accuracy <- rbind(vld_accuracy,
                              data.frame(n=i, k=k, acc=mean(val_data$class == tmp_vd_pred)))

    }
}

# Promedio por valor de k
vld_accuracy <- vld_accuracy %>% group_by(k) %>%
                    summarise(mean_acc=mean(acc))

# Veo la mejor configuración
vld_accuracy[which.max(vld_accuracy$mean_acc),]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 3 ##

# Ejemplo de k-fold cv
vld_accuracy <- data.frame()

folds <- 5

indexes <- sample(rep_len((1:folds), nrow(data_set)))

for (k in k_vals) {

    for (f in c(1:folds)) {

        print(c(k, f))

        train_data <- data_set[indexes!=f,]
        val_data  <- data_set[indexes==f,]

        tmp_vd_pred <- knn(train_data[,-ncol(train_data)],
                           val_data[,-ncol(val_data)], train_data$class, k)

        vld_accuracy  <- rbind(vld_accuracy,
                               data.frame(fold=f, k=k, acc=mean(val_data$class == tmp_vd_pred)))

    }
}

# Promedio por valor de k
vld_accuracy <- vld_accuracy %>% group_by(k) %>%
                    summarise(mean_acc=mean(acc))

# Veo la mejor configuración
vld_accuracy[which.max(vld_accuracy$mean_acc),]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 4 ##

# Ejemplo de training, validation y testing

test_indexes <- sample(c(1:nrow(data_set)), 280)  # Aproximadamente un 20% de las obs.
val_indexes <- sample(setdiff(c(1:nrow(data_set)), test_indexes), 280)

val_data <- data_set[val_indexes,]
test_data <- data_set[test_indexes,]
train_data <- data_set[-c(val_indexes, test_indexes),]

# Entreno con distintos valores K y valido
vld_accuracy <- data.frame()

for (k in k_vals) {

    print(k)

    tmp_vd_pred <- knn(train_data[,-ncol(train_data)],
                       val_data[,-ncol(val_data)], train_data$class, k)

    vld_accuracy <- rbind(vld_accuracy,
                          data.frame(k=k, acc=mean(val_data$class == tmp_vd_pred)))

}

# Guardo la mejor configuración
best_conf <- vld_accuracy[which.max(vld_accuracy$acc),]

# Modelo final (se entrena con todos los datos)
all_train_data <- rbind(train_data, val_data)

ts_pred <- knn(all_train_data[,-ncol(train_data)], test_data[,-ncol(test_data)],
               all_train_data$class, best_conf$k)

mean(test_data$class == ts_pred)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Bloque 5 ##

library(rpart)
library(rpart.plot)

# Separo en training y testing
val_data <- data_set[holdout_index,]  # El mismo holdout index que usé en el primer bloque
train_data <- data_set[-holdout_index,]

# Entreno un modelo de vecinos más cercanos
tree_fit <- rpart(class ~ ., data = train_data)  # Sintáxis similar a bayes ingenuo

# Veo cómo fue sobre los datos de testing
tree_predictions <- predict(tree_fit, newdata = val_data, type = "class")
print(mean(val_data$class == tree_predictions))

#Visualicemos el árbol
rpart.plot(tree_fit)

# Veamos la importancia de atributos
tree_fit$variable.importance

# Probemos distintas combinaciones de hiperparámetros

depths <- c(1, 2, 3, 5, 10, 20, 30)
minsplits <- c(1, 3, 5, 10, 15)
minbuckets <- c(1, 3, 5, 10, 15)

vld_accuracy <- data.frame()

for (ms in minsplits) {

    for (mb in minbuckets) {

        for (d in depths) {

            print(c(ms, mb, d))

            tree_fit <- rpart(class ~ ., data = train_data,
                              control = rpart.control(maxdepth = d, xval=0,
                                                      minsplit = ms, minbucket=mb, cp=0))

            tmp_vd_pred <- predict(tree_fit, val_data, type="class")

            vld_accuracy  <- rbind(vld_accuracy,
                                   data.frame(maxdepth = d,
                                              minsplit = ms,
                                              minbucket = mb,
                                              acc=mean(val_data$class == tmp_vd_pred)))
        }
    }
}

# Veo la mejor configuración
best_conf <- vld_accuracy[which.max(vld_accuracy$acc),]
print(best_conf)

# Entreno la mejor configuración con todos los datos
best_tree <- rpart(class ~ .,
                   data = train_data,
                   control = rpart.control(maxdepth = best_conf$maxdepth,
                                           minsplit = best_conf$minsplit,
                                           minbucket = best_conf$minbucket,
                                           xval=0, cp=0))

rpart.plot(best_tree)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#