setwd("C:/Users/Install/Desktop/Tree in R")
library(Metrics)
########################################################################
#Compare all models based on AUC
########################################################################
#we will perform a model comparison across all types of models that we've learned about so far
#Decision Trees
#Bagged Trees
#Random Forest
#Gradient Boosting Machine
#The models were all trained on the same training set
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
#Retrieve the prediction from the four models we studied
########################################################################
dt_preds <- readRDS("dt_preds")
bag_preds <- readRDS("bag_preds")
rf_preds <- readRDS("rf_preds")
gbm_preds <- readRDS("gbm_preds")

########################################################################
#Compare all models based on AUC
########################################################################

# Generate the test set AUCs using the two sets of predictions & compare
actual <- ifelse(credit_test$default == "yes", 1, 0)
dt_auc <- auc(actual = actual, predicted = dt_preds)
bag_auc <- auc(actual = actual, predicted = bag_preds)
rf_auc <- auc(actual = actual, predicted = rf_preds)
gbm_auc <- auc(actual = actual, predicted = gbm_preds)

# Print results
sprintf("Decision Tree Test AUC: %.3f", dt_auc)
sprintf("Bagged Trees Test AUC: %.3f", bag_auc)
sprintf("Random Forest Test AUC: %.3f", rf_auc)
sprintf("GBM Test AUC: %.3f", gbm_auc)

#Random Forest performed the best on the test set
#a bit more tuning of the GBM, the performance might be closer to that of the Random Forest

########################################################################
#Plot & compare ROC curves
########################################################################
#We conclude this course by plotting the ROC curves for all the models 
# The ROCR package provides the prediction() and performance() functions which generate the data required for plotting the ROC curve

#The more "up and to the left" the ROC curve of a model is, the better the model.
library(ROCR)
# List of predictions
preds_list <- list(dt_preds, bag_preds, rf_preds, gbm_preds)

# List of actual values (same for all)
m <- length(preds_list)
actuals_list <- rep(list(credit_test$default), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Decision Tree", "Bagged Trees", "Random Forest", "GBM"),
       fill = 1:m)
