library(factoextra) 
library(dplyr)
diabetes<- read.csv("diabetes.csv", header = TRUE, sep=",")
head(diabetes)
data <- diabetes[,1:8]
outcome <- diabetes[,9]
head(data)
head(outcome)
data_standardized  <- scale(x = data)
data_covariance <- cov(data_standardized)
data_eigen <- eigen(data_covariance)
data_eigen
data_pca <- prcomp(x = data, scale. = TRUE, center = TRUE)
names(data_pca)
summary(data_pca)
data_pca$rotation
plot_pca <- plot(data_pca, type="l")
plot_pca

