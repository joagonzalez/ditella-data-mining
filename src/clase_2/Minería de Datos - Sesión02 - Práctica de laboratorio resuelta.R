rm(list = ls())

library(rpart)
library(dplyr)
library(ggplot2)

setwd("../Downloads/")

# Cargo los datos
# Esto es un poco tedioso, porque el separador en ", " (coma con un espacio depsués, y estoy haciendo que
# lo lea súper prolijo, read.table con "," como separado debería andar bien.)
# OJO que los NA están como "?".

train_data <- read.table(text = gsub(", ", ",", readLines("adult.data")),
                         sep = ",", header = FALSE, na.strings = "?")

test_data <- read.table(text = gsub("\\.$", "", gsub(", ", ",", readLines("adult.test")[-1])),
                        sep = ",", header = FALSE, na.strings = "?")

adult_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country", "income")

colnames(train_data) <- adult_names
colnames(test_data) <- adult_names

# Quito native_country de train porque trae problemas
train_data <- train_data %>% select(-native_country)

# Ejemplo de k-fold cv
vld_accuracy <- data.frame()

folds <- 5

indexes <- sample(rep_len((1:folds), nrow(train_data)))

cp_vals <- seq(from = 0, to = 1, by = 0.01)

for (cp in cp_vals) {
  
  for (f in c(1:folds)) {
    
    print(c(f, cp))
    
    train_data_tmp <- train_data[indexes!=f,]
    val_data_tmp  <- train_data[indexes==f,]
    
    tree_fit <- rpart(income ~ ., data = train_data_tmp,
                      control = rpart.control(maxdepth = 30, xval=0,
                                              minsplit = 1, minbucket=1, cp = cp))
    
    tmp_vd_pred <- predict(tree_fit, val_data_tmp, type = "class")
    
    vld_accuracy  <- rbind(vld_accuracy,
                           data.frame(fold = f, cp = cp,
                                      acc = mean(val_data_tmp$income == tmp_vd_pred)))
    
  }
}

# Promedio los folds por valor de cp
vld_accuracy <- vld_accuracy %>% group_by(cp) %>%
  summarise(mean_acc=mean(acc), sd_acc=sd(acc))

# Grafico la evolución de accuracy
ggplot(vld_accuracy, aes(x = cp, y = mean_acc)) + geom_line() + theme_minimal()

# Guardo la mejor configuación (de todos los cp que dan el máximo, me quedo con el más grande)
best_cp <- vld_accuracy[which.max(vld_accuracy$mean_acc),]

# Final tree
final_tree_fit <- rpart(income ~ ., data = train_data,
                        control = rpart.control(maxdepth = 30, xval=0,
                                                minsplit = 1, minbucket=1,
                                                cp = best_cp$cp))

# Perfomrance del árbol final en testeo
mean(predict(final_tree_fit, newdata = test_data, type = "class") == test_data$income)
