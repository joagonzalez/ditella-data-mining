rm(list=ls())
setwd('/home/jgonzalez/Downloads/data')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 1er bloque

library('Matrix')

M1 <- matrix(0, nrow = 1000, ncol = 1000)
M2 <- Matrix(0, nrow = 1000, ncol = 1000, sparse = TRUE)

format(object.size(M1), unit = "Mb")
format(object.size(M2), unit = "Mb")

M1[500, 500] <- 1
M2[500, 500] <- 1
 
format(object.size(M1), unit = "Mb")
format(object.size(M2), unit = "Mb")

M1 <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
M2 <- Matrix(rnorm(1000000), nrow = 1000, ncol = 1000, sparse = TRUE)

format(object.size(M1), units= "Mb")
format(object.size(M2), units= "Mb")

M1 <- sparseMatrix(i=c(2, 20), j=c(3, 5), x=c(4, 8))
class(M1)
print(M1)

M2 <- sparseMatrix(i=c(2, 20), j=c(3, 5), x=c(4, 8), dims = c(25, 10))
print(M2)

M2 <- sparseMatrix(i=c(2, 20), j=c(3, 5), x=c(4, 8), dims = c(25000, 10000))
print(M2)

format(object.size(M1), units= "Mb")
format(object.size(M2), units= "Mb")

rm(M1, M2)
gc()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 2do bloque
library(data.table)

load_csv_data <- function(csv_file, sample_ratio = 1, drop_cols = NULL,
                          sel_cols = NULL) {

    dt <- fread(csv_file, header = TRUE, sep = ",", stringsAsFactors = TRUE,
                na.strings = "", drop = drop_cols, select = sel_cols,
                showProgress = TRUE)

    if (sample_ratio < 1) {
        sample_size <- as.integer(sample_ratio * nrow(dt))
        dt <- dt[sample(.N, sample_size)]
    }

    return(dt)
}


one_hot_sparse <- function(data_set) {

    require(Matrix)

    created <- FALSE

    if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Label"  es numérica y tiene que estar sí o sí)
        out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric), with = FALSE]), "dgCMatrix")  # Si en lugar de pasar un objeto de data table, pasan un data.frame común no debe decir ", with = FALSE" en ninguna instrucción
        created <- TRUE
    }

    if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
        if (created) {
            out_put_data <- cbind2(out_put_data,
                                    as(as.matrix(data_set[,sapply(data_set, is.logical),
                                                           with = FALSE]), "dgCMatrix"))
        } else {
            out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical), with = FALSE]), "dgCMatrix")
            created <- TRUE
        }
    }

    # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
    fact_variables <- names(which(sapply(data_set, is.factor)))

    # Para cada columna factor hago one hot encoding
    i <- 0

    for (f_var in fact_variables) {

        f_col_names <- levels(data_set[[f_var]])
        f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
        j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
        
        if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
            j_values[is.na(j_values)] <- length(f_col_names) + 1
            f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
        }

        if (i == 0) {
            fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                      x = rep(1, nrow(data_set)),
                                      dims = c(nrow(data_set), length(f_col_names)))
            fact_data@Dimnames[[2]] <- f_col_names
        } else {
            fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                          x = rep(1, nrow(data_set)),
                                          dims = c(nrow(data_set), length(f_col_names)))
            fact_data_tmp@Dimnames[[2]] <- f_col_names
            fact_data <- cbind(fact_data, fact_data_tmp)
        }

        i <- i + 1
    }

    if (length(fact_variables) > 0) {
        if (created) {
            out_put_data <- cbind(out_put_data, fact_data)
        } else {
            out_put_data <- fact_data
            created <- TRUE
        }
    }
    return(out_put_data)
}


## Defino que variables voy a cargar
TO_KEEP <- c("platform", "age", "install_date", "id",
             "TutorialStart", "Label_max_played_dsi",
             "StartSession_sum_dsi0", "StartSession_sum_dsi1",
             "StartSession_sum_dsi2", "StartSession_sum_dsi3",
             "categorical_1", "categorical_2", "categorical_3",
             "categorical_4", "categorical_5", "categorical_6",
             "categorical_7", "device_model")

## Cargo uno de los datasets de entrenamiento del tp
train_set <- load_csv_data("train_3.csv",
                           sample_ratio = 0.25,
                           sel_cols = TO_KEEP)

train_set[, train_sample := TRUE]

## Cargo el dataset de evaluación del TP
eval_set <- load_csv_data("evaluation.csv",
                          sel_cols = setdiff(TO_KEEP,
                                             "Label_max_played_dsi"))

eval_set[, train_sample := FALSE]

## Uno los datasets
data_set <- rbind(train_set, eval_set, fill = TRUE)
rm(train_set, eval_set)
gc()

# Inspecciono la estructura de los datos
str(data_set)

## Hago algo de ingeniería de atributos
data_set[, Label := as.numeric(Label_max_played_dsi == 3)]
data_set[, Label_max_played_dsi := NULL]

data_set[, max_StartSession_sum := pmax(StartSession_sum_dsi0,
                                        StartSession_sum_dsi1,
                                        StartSession_sum_dsi2,
                                        StartSession_sum_dsi3,
                                        na.rm = TRUE)]

data_set[, min_StartSession_sum := pmin(StartSession_sum_dsi0,
                                        StartSession_sum_dsi1,
                                        StartSession_sum_dsi2,
                                        StartSession_sum_dsi3,
                                        na.rm = TRUE)]

## Hago one hot encoding
data_set <- one_hot_sparse(data_set)
gc()

dim(data_set)
colnames(data_set)

## Separo en conjunto de training y evaluación de nuevo
train_set <- data_set[as.logical(data_set[,"train_sample"]),]
train_set <- train_set[, setdiff(colnames(train_set), "train_sample")]
eval_set <- data_set[!as.logical(data_set[,"train_sample"]),]
eval_set <- eval_set[, setdiff(colnames(eval_set), "train_sample")]
rm(data_set)
gc()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 3er bloque

library(xgboost)

val_index <- c(1:10000)

train_index <- setdiff(c(1:nrow(train_set)), val_index)

dtrain <- xgb.DMatrix(data = train_set[train_index,
                                       colnames(train_set) != "Label"],
                      label = train_set[train_index,
                                        colnames(train_set) == "Label"])

dvalid <- xgb.DMatrix(data = train_set[val_index, colnames(train_set) != "Label"],
                      label = train_set[val_index, colnames(train_set) == "Label"])

## Entreno un modelo de xgboost con watchlist
watchlist <- list(train = dtrain, valid = dvalid)

vanilla_model <- xgb.train(data = dtrain, nrounds = 40,
                           watchlist = watchlist,
                           objective = "binary:logistic",  # Es la función objetivo para clasificación binaria
                           eval.metric = "auc",
                           print_every_n = 5)

## Examino la importancia de variables según xgboost
head(xgb.importance(model=vanilla_model), 20)  # Ver https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html

# Predecimos sobre los datos de evaluación
eval_preds <- data.frame(id = eval_set[, "id"],
                         Label = predict(vanilla_model,
                                         newdata = eval_set[,setdiff(colnames(eval_set), "Label")]))

# Armo el archivo para subir a Kaggle
options(scipen = 999)  # Para evitar que se guarden valores en formato científico
write.table(eval_preds, "modelo_ya_menos_basico.csv",
            sep = ",", row.names = FALSE, quote = FALSE)
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 4to bloque
library(MASS)  # Sólo para tener el dataset Boston disponible
library(rpart)
library(rpart.plot)

#Dataset con el que vamos a trabajar
head(Boston)
str(Boston)
summary(Boston)

# Veamos parte de los datos
plot(Boston[,c(1:5)])

# Prueba de K-means
clusters <- kmeans(Boston, centers=7, iter.max=30,  nstart=20)
clusters$centers
clusters$cluster
table(clusters$cluster)

# Veamos con evoluciona la función objetivo a medida que aumenta K
evol_variabilidad <- data.frame()
for (i in c(1:20)) {
    clusters <-kmeans(Boston, centers=i, iter.max=30,  nstart=20)
    evol_variabilidad <- rbind(evol_variabilidad,
                               data.frame(k=i,
                                          var=clusters$tot.withinss))
}

plot(c(1:20), evol_variabilidad$var, type="o",
     xlab="# Clusters", ylab="tot.withinss")

# Veamos cómo se asignaron los clusters
clusters <- kmeans(Boston, centers=4, iter.max=30,  nstart=20)
plot(Boston, col=clusters$cluster)

# Interpretemos los clusters
Boston$cluster <- factor(clusters$cluster)
rpart.plot(rpart(cluster ~ ., data = Boston, control = list(maxdepth = 4)))
plot(Boston[,c("tax", "black")], col=clusters$cluster)

# Noten qué característica tienen tax y black
summary(Boston)
# Prueben rehacer el análisis pero tomando z-scores de las variables