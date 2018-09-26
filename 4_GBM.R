setwd("C:/Users/Install/Desktop/Tree in R")

########################################################################
#Definition Gradient Boosting Machine (G.B.M)
########################################################################
#QUESTION : What is the main difference between bagged trees and boosted trees?
#RESPONSE : Boosted trees improve the model fit by considering past fits and bagged trees do not.
#COMMENT :  Boosting is an iterative algorithm that considers past fits to improve performance.
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
#Train a GBM model
########################################################################
#gbm() function to train a GBM classifier to predict loan default
#You will train a 10,000-tree GBM on the credit_train

#Using such a large number of trees (10,000) is probably not optimal for a GBM 
#but we will build more trees than we need and then select the optimal number of trees based on early performance-based stopping. 

#For binary classification, gbm() requires the response to be encoded as 0/1 (numeric)
#so we will have to convert from a "no/yes" factor to a 0/1 numeric 

#gbm() function requires the user to specify a distribution argument
#For a binary classification problem, you should set distribution = "bernoulli"

# Convert "yes" to 1, "no" to 0
credit_train$default <- ifelse(credit_train$default == "yes", 1, 0)

library(gbm)
# Train a 10000-tree GBM model
set.seed(1)
credit_model <- gbm(formula = default ~ ., 
                    distribution = "bernoulli", 
                    data = credit_train,
                    n.trees = 10000)

# Print the model object                    
print(credit_model)

# summary() prints variable importance
summary(credit_model)

########################################################################
#Prediction using a GBM model
########################################################################
# Since we converted the training response col, let's also convert the test response col
credit_test$default <- ifelse(credit_test$default == "yes", 1, 0)

# Generate predictions on the test set
preds1 <- predict.gbm(object = credit_model, 
                  newdata = credit_test,
                  n.trees = 10000)

# Generate predictions on the test set (scale to response)
preds2 <- predict.gbm(object = credit_model, 
                  newdata = credit_test,
                  n.trees = 10000,
                  type = "response")

# Compare the range of the two sets of predictions
range(preds1)
range(preds2)

########################################################################
#Evaluate test set AUC
########################################################################
#Compute test set AUC of the GBM model for the two sets of predictions
#We will notice that they are the same value.
#AUC is a rank-based metric, so changing the actual values does not change the value of the AUC.
# Generate the test set AUCs using the two sets of preditions & compare
auc(actual = credit_test$default, predicted = preds1)  #default
auc(actual = credit_test$default, predicted = preds2)  #rescaled 

########################################################################
#Early stopping in GBMs
########################################################################
#Use the gbm.perf() function to estimate the optimal number of boosting 
#(aka n.trees)
#using both OOB and CV error

#When you set out to train a large number of trees in a GBM (such as 10,000) 
#you use a validation method to determine an earlier (smaller) number of trees, then that's called "early stopping"

# Optimal ntree estimate based on OOB
ntree_opt_oob <- gbm.perf(object = credit_model, 
                          method = "OOB", 
                          oobag.curve = TRUE)
#get the optimal number of trees based on the OOB error and store that number
#print(ntree_opt_oob)
#value = 3233

# Train a CV GBM model
set.seed(1)
credit_model_cv <- gbm(formula = default ~ ., 
                       distribution = "bernoulli", 
                       data = credit_train,
                       n.trees = 10000,
                       cv.folds = 2)

# Optimal ntree estimate based on CV
ntree_opt_cv <- gbm.perf(object = credit_model_cv, 
                         method = "cv")

# Compare the estimates                         
print(paste0("Optimal n.trees (OOB Estimate): ", ntree_opt_oob))                         
print(paste0("Optimal n.trees (CV Estimate): ", ntree_opt_cv))

########################################################################
#OOB vs CV-based early stopping
########################################################################
#Between OOB and CV compare the performance of the models on a test set
# Generate predictions on the test set using ntree_opt_oob number of trees
preds1 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = ntree_opt_oob)

# Generate predictions on the test set using ntree_opt_cv number of trees
preds2 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = ntree_opt_cv)   
#saveRDS(preds2, file = "gbm_preds")

library(Metrics)
# Generate the test set AUCs using the two sets of preditions & compare
auc1 <- auc(actual = credit_test$default, predicted = preds1)  #OOB
auc2 <- auc(actual = credit_test$default, predicted = preds2)  #CV 

# Compare AUC 
print(paste0("Test set AUC (OOB): ", auc1))                         
print(paste0("Test set AUC (CV): ", auc2))

#Cross-validation's early stop is slightly better than OOB's error.

########################################################################
#Best GBM model
########################################################################
