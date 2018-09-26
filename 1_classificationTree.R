#setwd("C:/Users/Install/Desktop/Tree in R")

########################################################################
#Build a classification tree
########################################################################
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
#Train a classification tree model
########################################################################
library(rpart)
library(rpart.plot)
# Train the model (to predict 'default')
credit_model <- rpart(formula = default ~., 
                      data = credit_train, 
                      method = "class")

# Look at the model output                      
print(credit_model)
# Display the results
rpart.plot(x = credit_model, yesno = 2, type = 0, extra = 0)
########################################################################
#Compute confusion matrix
########################################################################
library(caret)
# Generate predicted classes using the model object
class_prediction <- predict(object = credit_model,  
                            newdata = credit_test,   
                            type = "class")  

# Calculate the confusion matrix for the test set
confusionMatrix(data = class_prediction,  
                reference = credit_test$default)  
#Accuracy 0.75 VS No info 0.69
########################################################################
#Compare models with a different splitting criterion
########################################################################
#Train two models that use a different splitting criterion
#use the validation set to choose a "best" model from this group
# Train a gini-based model
credit_model1 <- rpart(formula = default ~ ., 
                       data = credit_train, 
                       method = "class",
                       parms = list(split = "gini"))
# Train an information-based model
credit_model2 <- rpart(formula = default ~ ., 
                       data = credit_train, 
                       method = "class",
                       parms = list(split = "information"))
# Generate predictions on the validation set using the gini model
pred1 <- predict(object = credit_model1, 
                 newdata = credit_test,
                 type = "class")    
# Generate predictions on the validation set using the information model
pred2 <- predict(object = credit_model2, 
                 newdata = credit_test,
                 type = "class")
library(Metrics)
# Compare classification error
ce(actual = credit_test$default, 
   predicted = pred1)
ce(actual = credit_test$default, 
   predicted = pred2)  
#Information parameter is better from -0.20 points

###################################################################################
#Prepare Prediction parameter "info" for model comparison
###################################################################################
# Generate predictions on the validation set using the information model
pred3 <- predict(object = credit_model2, 
                 newdata = credit_test,
                 type = "prob")
#saveRDS(pred3[,"yes"],file="dt_preds")