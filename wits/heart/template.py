# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:24:58 2021

Heart Disease Data - Machine Learning Classification

@author: john.atherfold
"""

#%% 0. Import the python libraries you think you'll require


#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')


#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)


#   2.2 Exploring the dependence in the predictors/inputs

# This particular feature is logged - spreads the data out a bit more, as this was
# a highly tailed distribution


#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.

# Split the Test Set stratified on our target class. This can be checked by plotting
# histograms of yTest and yTrainValid and comparing them. They should look the same.

# Balance the data set by undersampling the major class (this can also be done
# by oversampling the minor class)


#Check for balance


# Extract categorical predictors. We'd like to scale the data and run a PCA on
# it, but this should not be done on binary/categorical data


#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

# There is a class imbalance in the data. Lets assume all data points belong to
# the same class as a start, and compare our models against that.


# The model has an accuracy of 67.9%, which may seem good, but it's actually just a random guess.
# Also it's noteworthy that the F1 score is 0. A good model has an F1 score close to 1

#%% 5. Identify a Suitable Machine Learning Model



    # Remove the categorical data


    # Create the scaler object to scale the data. Fit scaler on Training data and
    # Apply scaler on both Training and Validation data. Prevent data leakage.


    # Create principal component from training data. Fit PCA on training data only,
    # and project validation data onto those PCs


    # Append categorical data back to continuous data



    # Fit the model and test accuracy





#%% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared

# Train final model on ALL training and validation data, then test on test data.
# Remember, the function of cross-validation was to choose hyperparameters for
# our machine learning model. In this case, our hyper parameters were the type
# of kernel we used in out SVM (poly or rbf), and the regularisation  value, C.
# These were chosen based on the results obtained from k-fold cross-validation.

# Remove the categorical data


# Create the scaler object to scale the data. Fit scaler on Training data and
# Apply scaler on both Training and Validation data. Prevent data leakage.


# Create principal component from training data. Fit PCA on training data only,
# and project validation data onto those PCs


# Append categorical data back to continuous data


#%% 7. Add Complexity if Required
# In this case, the complexity is added in the functionality, and is iterated
# upon by uncommenting the appropriate model-fitting lines

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question
