setwd("C:/Users/Install/Desktop/Tree in R")

########################################################################
#Definition Random Forest
########################################################################
#What is the main difference between bagged trees and the Random Forest algorithm?
#In Random Forest, only a subset of features are selected at random at each split in a decision tree. 
#In bagging, all features are used.
credit <- read.csv("credit.csv", stringsAsFactors = TRUE)
########################################################################
#Split data in 80% 20%
########################################################################
# Total number of rows in the credit data frame
n <- nrow(credit)
# Number of rows for the training set (80% of the dataset)
n_train <- round(0.8 * n) 
# Create a vector of indices which is an 80% random sample
set.seed(123)
train_indices <- sample(1:n, n_train)
# Subset the credit data frame to training indices only
credit_train <- credit[train_indices, ]  
# Exclude the training indices to create the test set
credit_test <- credit[-train_indices, ]

########################################################################
#Train a Random Forest
########################################################################
library(randomForest)
# Train a Random Forest
set.seed(1)  # for reproducibility
credit_model <- randomForest(formula = default ~ ., 
                             data = credit_train)

# Print the model output                             
print(credit_model)

########################################################################
#Evaluate out-of-bag error
########################################################################
#plot the OOB error as a function of the number of trees trained
#extract the final OOB error of the Random Forest model from the trained model object.
# Grab OOB error matrix & take a look
err <- credit_model$err.rate
head(err)

# Look at final OOB error rate (last row in err matrix)
oob_err <- err[nrow(err), "OOB"]
print(oob_err)

# Plot the model trained in the previous exercise
plot(credit_model)

# Add a legend since it doesn't have one by default
legend(x = "right", 
       legend = colnames(err),
       fill = 1:ncol(err))

########################################################################
#Evaluate model performance on a test set
########################################################################
#Use the caret::confusionMatrix() function to compute test set accuracy 
#Compare the test set accuracy to the OOB accuracy.
library(caret)
# Generate predicted classes using the model object
class_prediction <- predict(object = credit_model,   # model object 
                            newdata = credit_test,  # test dataset
                            type = "class") # return classification labels

# Calculate the confusion matrix for the test set
cm <- confusionMatrix(data = class_prediction,       # predicted classes
                      reference = credit_test$default)  # actual classes
print(cm)
#Accuracy = 0.755

# Compare test set accuracy to OOB accuracy
paste0("Test Accuracy: ", cm$overall[1])
paste0("OOB Accuracy: ", 1 - oob_err)  

########################################################################
#Advantage of OOB error
########################################################################
#QUESTION : What is the main advantage of using OOB error instead of validation or test error?
#REPSONSE : If you evaluate your model using OOB error, then you don't need to create a separate test set.
#COMMENT : This allows you to use all of rows in your original dataset for training

########################################################################
#Evaluate test set AUC
########################################################################
# Generate predictions on the test set
pred <- predict(object = credit_model,
                newdata = credit_test,
                type = "prob")

# `pred` is a matrix
class(pred)

# Look at the pred format
head(pred)

library(Metrics)
# Compute the AUC (`actual` must be a binary 1/0 numeric vector)
auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = pred[,"yes"])
#AUC = 0.7989

########################################################################
#Tuning a Random Forest via mtry
########################################################################
#randomForest::tuneRF() to tune mtry (by training several models).

#This function is a specific utility to tune the mtry parameter based on OOB error
#which is helpful when you want a quick & easy way to tune your model
#ntreeTry that defaults to 50 
# Execute the tuning process
set.seed(1)              
res <- tuneRF(x = subset(credit_train, select = -default),
              y = credit_train$default,
              ntreeTry = 500)

# Look at results
print(res)

# Find the mtry value that minimizes OOB Error
mtry_opt <- res[,"mtry"][which.min(res[,"OOBError"])]
print(mtry_opt)

# If you just want to return the best RF model (rather than results)
# you can set `doBest = TRUE` in `tuneRF()` to return the best RF model
# instead of a set performance matrix.

#Mtry which minimize the OOB error (0.23) is 4 mtry.

########################################################################
#Tuning a Random Forest via tree depth
########################################################################
#In Chapter 2, we created a manual grid of hyperparameters using the expand.grid()
#In this exercise, you will create a grid of mtry, nodesize and sampsize values

#In this example, we will identify the "best model" based on OOB error.
#The best model is defined as the model from our grid which minimizes OOB error.

#Keep in mind that there are other ways to select a best model from a grid, such as choosing the best model based on validation AUC

# Establish a list of possible values for mtry, nodesize and sampsize
mtry <- seq(4, ncol(credit_train) * 0.8, 2)
nodesize <- seq(3, 8, 2)
sampsize <- nrow(credit_train) * c(0.7, 0.8)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model <- randomForest(formula = default ~ ., 
                        data = credit_train,
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  
  # Store OOB error for the model                      
  oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])

########################################################################
#Evaluate BEST MODEL based on tuning hyperparamters on OOB erros
########################################################################

#Train Best Random Forest
Best_model <- randomForest(formula = default ~.,
                        data = credit_train,
                        mtry = 10,
                        nodesize = 3,
                        sampsize = 640)
Best_pred <- predict(object = Best_model,
                newdata = credit_test,
                type = "prob")

#Saved Positive prediction
saveRDS(Best_pred[,"yes"],file = "rf_preds")


# Compute the AUC (`actual` must be a binary 1/0 numeric vector)
auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = Best_pred[,"yes"])
#AUC = 0.81121

