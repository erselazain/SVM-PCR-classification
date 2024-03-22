library(caTools)
library(e1071)
library(caret)
library(scatterplot3d)
library(factoextra) 
library(dplyr)

# Masukkan data
dataset <- read.csv("diabetes.csv")

# Encoding the target feature as factor 
dataset$Outcome <- as.factor(dataset$Outcome)

# Splitting the dataset into the Training set and Test set 
set.seed(123)
split = sample.split(dataset$Outcome, SplitRatio = 0.80) 

training_set = subset(dataset, split == TRUE) 
test_set = subset(dataset, split == FALSE) 

# Feature Scaling 
training_set[-9] = scale(training_set[-9])
test_set[-9] = scale(test_set[-9]) 

# Membangun model dengan kernel berbeda
model_linear <- svm(formula = Outcome ~ ., data = training_set, 
                    kernel = "linear")
model_radial <- svm(formula = Outcome ~ ., data = training_set, 
                    kernel = "radial", gamma = 0.01)
model_poly2 <- svm(formula = Outcome ~ ., data = training_set, 
                  kernel = "poly", degree = 3, coef0 = 0, gamma = 10)
model_poly3 <- svm(formula = Outcome ~ ., data = training_set, 
                   kernel = "poly", degree = 3, coef0 = 0, gamma = 10)
model_sigmoid <- svm(formula = Outcome ~ ., data = training_set, 
                     kernel = "sigmoid", gamma = 0.1, coef0 = 0)

# Membuat data aktual dari target data tes
actual <- test_set$Outcome

# Prediksi dan membandingkan dengan data tes
pred_linear <- predict(model_linear, test_set, decision.values = TRUE)
cm_linear <- table(actual, pred_linear)
cm_linear
acc_linear <- sum(diag(cm_linear)) / sum(cm_linear)
print(paste('Akurasi SVM Linear', acc_linear * 100, '%'))

pred_radial <- predict(model_radial, test_set, decision.values = TRUE)
cm_radial <- table(actual, pred_radial)
cm_radial
acc_radial <- sum(diag(cm_radial)) / sum(cm_radial)
print(paste('Akurasi SVM Radial', acc_radial * 100, '%'))

pred_poly2 <- predict(model_poly2, test_set, decision.values = TRUE)
cm_poly2 <- table(actual, pred_poly2)
cm_poly2
acc_poly2 <- sum(diag(cm_poly2)) / sum(cm_poly2)
print(paste('Akurasi SVM Polynomial', acc_poly2 * 100, '%'))

pred_poly3 <- predict(model_poly3, test_set, decision.values = TRUE)
cm_poly3 <- table(actual, pred_poly3)
cm_poly3
acc_poly3 <- sum(diag(cm_poly3)) / sum(cm_poly3)
print(paste('Akurasi SVM Polynomial', acc_poly3 * 100, '%'))

pred_sigmoid <- predict(model_sigmoid, test_set, decision.values = TRUE)
cm_sigmoid <- table(actual, pred_sigmoid)
cm_sigmoid
acc_sigmoid <- sum(diag(cm_sigmoid)) / sum(cm_sigmoid)
print(paste('Akurasi SVM Sigmoid', acc_sigmoid * 100, '%'))

#Reduksi Dimensi dengan PCA
pca = preProcess(training_set[,-9], method='pca', pcaComp = 8)
pca$rotation
print(pca)

train = predict(pca, training_set)
head(train)# column orders change. The DV becomes the first variable
test =  predict(pca, test_set)
head(test)

# Just rearranging the columns
train = train[,c(2,3,1)]
head(train)
test  = test[,c(2,3,1)]

# Create a logistic rgression model on the reduced data
modSVM = svm(Outcome~.,data=train,
             type='C-classification',
             kernel='radial')

# Predict the outcomes using the model
head(test[,-3])
y_pred = predict(modSVM, test[,-3])
head(y_pred)

# Visualize the decision boundaries for training set
set = train

X1 = seq(from=min(set[,1])-1, to=max(set[,1]+1), by=0.02)
X2 = seq(from=min(set[,2])-1, to=max(set[,2]+1), by=0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(modSVM, grid_set)

plot(set[,-3],
     main = 'SVM after PCA (Data Train)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))

#contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)   # this is optional

points(grid_set, pch='.',col=ifelse(y_grid==2,'deepskyblue',ifelse(y_grid==1,'springgreen3','tomato')))
points(set, pch=21, bg=ifelse(set[,3]==2,'blue3', ifelse(set[,3]==1, 'green4','red3')))

# Visualize the decision boundaries for test set
set = test

X1 = seq(from=min(set[,1])-1, to=max(set[,1]+1), by=0.02)
X2 = seq(from=min(set[,2])-1, to=max(set[,2]+1), by=0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(modSVM, grid_set)

plot(set[,-3],
     main = 'SVM after PCA (Data Test)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))

#contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)   # this is optional

points(grid_set, pch='.',col=ifelse(y_grid==2,'deepskyblue',ifelse(y_grid==1,'springgreen3','tomato')))
points(set, pch=21, bg=ifelse(set[,3]==2,'blue3', ifelse(set[,3]==1, 'green4','red3')))

