rm(list=ls())
setwd('/home/jgonzalez/Downloads/data')

library(arules)
library(caret)
library(dplyr)
library(readxl)
library(stringr)
library(tm)

retail <- read_excel("Online Retail.xlsx")
head(retail, 20)

# Debo pasar el identificador de transacción y el de producto a factores
retail <- retail %>%
  select(InvoiceNo, Description) %>% distinct() %>%
  filter(complete.cases(.)) %>%
  mutate(InvoiceNo = as.factor(InvoiceNo), Description = as.factor(Description))
head(retail, 20)

retail <- split(retail$Description, retail$InvoiceNo)
head(retail)

baskets <- as(retail, "transactions")
rm(retail)

summary(baskets)

association.rules <- apriori(baskets, parameter = list(supp=0.005, conf=0.5, maxlen=4))

summary(association.rules)

df_rules <- DATAFRAME(association.rules)

# Inspecciono las reglas
df_rules %>% filter(confidence > 0.8) %>% arrange(desc(lift)) %>% head(10)

rm(baskets, association.rules, df_rules)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Construcción de una dt.mat

puntaje_review <- read.table("clasificacion.txt", sep = "\t", header = TRUE)
head(puntaje_review)

corpus <- VCorpus(DirSource("corpus", encoding = "UTF-8"), readerControl = list(language="es"))  # Cargo el corpus
length(corpus)

# Usuarios de windows usen: corpus <- VCorpus(DirSource("corpus", encoding = "windows-1251"), readerControl = list(language = "es"))
corpus[[45]]
corpus[[45]]$meta
corpus[[45]]$content

# Paso a minúscuola
corpus <- tm_map(corpus, content_transformer(tolower))
corpus[[45]]$content

# Quito signos de puntuación
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus[[45]]$content

# Quito stopwords
print(stopwords("spanish"))
stp_words <- stopwords("spanish")[stopwords("spanish") != "no"]
corpus <- tm_map(corpus, content_transformer(function(x) removeWords(x, stp_words)))
corpus[[45]]$content

# Quito doble espacio y espacio al final (esto es de TOC)
corpus <- tm_map(corpus, content_transformer(function(x) gsub("\\s+", " ", str_trim(x))))
corpus[[45]]$content

# Pruebo el tokenizador
scan_tokenizer(corpus[[45]]$content)

# Paso a document-term matrix, no considero palabras que aparecen menos de 20 veces
dt.mat <- as.matrix(DocumentTermMatrix(corpus,
                                       control = list(stopwords = FALSE,
                                                      wordLengths = c(1, Inf),
                                                      bounds = list(global=c(15,Inf)))))
rm(corpus)
gc()

# Elimino post que quedaron sin palabras (esto no es necesario, pero lo hago para evitar problemas raros)
dt.mat <- dt.mat[rowSums(dt.mat) != 0,]
dim(dt.mat)

# Asigno una columna que sea id de comentario
dt.mat <- data.frame(id_comentario = gsub(".txt", "", rownames(dt.mat)), dt.mat)
rows.dt.mat <- rownames(dt.mat)

# Uno la clase y separo en "excelente" y "no_excelente"
dt.mat <- merge(dt.mat, puntaje_review, by = "id_comentario")  # Inner join
table(dt.mat$clase_comentario)
dt.mat$clase_comentario <- factor(ifelse(dt.mat$clase_comentario == "Malo", "malo", "no_malo"))

head(dt.mat[, c("pizza", "zona", "rico", "feo", "clase_comentario")], 300)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Clasificación de reviews

# Entreno un modelo que detecte sentimiento
fitControl <- trainControl(method = "LGOCV", number = 1, p = 0.75,
                           verboseIter = TRUE, classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# xgboost
xgbTreeFit <- train(x = dt.mat %>% select(-clase_comentario, -id_comentario), y = dt.mat$clase_comentario,
                    method = "xgbTree",
                    trControl = fitControl,
                    tuneGrid = data.frame(eta=0.1, max_depth=3,
                                          gamma=0.25, colsample_bytree=0.8,
                                          min_child_weight=1,
                                          subsample=0.75, nrounds=150),
                    metric = "ROC")

xgbTreeFit$results[which.max(xgbTreeFit$results$ROC),]

plot(varImp(xgbTreeFit), top = 30)

rm(fitControl, xgbTreeFit)
gc()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Clustering de reviews

# Muestreo 1500 documentos al azar y calculo la distancia coseno
dt.mat.dist <- as.matrix(dt.mat[sample(c(1:nrow(dt.mat)), 1500),] %>%
                           select(-clase_comentario, -id_comentario))

dt.mat.dist <- t(apply(dt.mat.dist, 1, function(x)(x/sqrt(sum(x^2)))))

dist.cos <- 1 - dt.mat.dist %*% t(dt.mat.dist)

# Hago clustering
hc.docs <- hclust(as.dist(dist.cos), method="complete")
plot(hc.docs, main="Cluster of documents", xlab="" , sub="" , cex=.9, labels=FALSE)

rm(hc.docs, dt.mat.dist, dist.cos)
gc()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Arules de reviews

# Creo nuevas columnas que van a cumplir el rol de nuevos items
dt.mat$clase_com_malo <- as.numeric(dt.mat$clase_comentario == "malo")
dt.mat$clase_com_no_malo <- as.numeric(dt.mat$clase_comentario == "no_malo")
dt.mat$clase_comentario <- NULL

# Creo las baskets (esta parte del código se podría optimizar, por ej: paralelizando)
id_comentarios <- as.character(dt.mat$id_comentario)
dt.mat$id_comentario <- NULL
dt.mat[,]  <- dt.mat > 0  # Lleno la matriz de valores lógicos (https://stackoverflow.com/questions/19569391/convert-character-matrix-to-logical)
dt.mat <- as.matrix(dt.mat)

baskets <- list()

for (i in seq_along(id_comentarios)) {
  if ((i %% 1000) == 0) {
    print(i)
  }
  baskets[[id_comentarios[i]]] <- names(which(dt.mat[i,]))
}

head(baskets, 5)

baskets <- as(baskets, "transactions")
rm(dt.mat)
gc()

# Encuentro las reglas de asociación
association.rules <- apriori(baskets, parameter = list(supp=0.001, conf=0.3))
arules_df <- DATAFRAME(association.rules)
arules_df <- arules_df %>% mutate(lhs_len = str_count(LHS, ",") + 1)

# Inspecciono algunas reglas que tienen como consecuente "malo" y "no_malo"
arules_df %>% filter(RHS == "{clase_com_no_malo}", count > 50) %>% arrange(desc(confidence)) %>% head(20)

arules_df %>% filter(RHS == "{clase_com_malo}", count > 50) %>% arrange(desc(confidence)) %>% head(20)

arules_df %>%filter(RHS == "{clase_com_no_malo}", lhs_len == 4, support > 0.004) %>%
  arrange(desc(lhs_len), desc(confidence)) %>% head(20)

arules_df %>%filter(RHS == "{clase_com_malo}", lhs_len == 4, support > 0.001) %>%
  arrange(desc(lhs_len), desc(confidence)) %>% head(20)
