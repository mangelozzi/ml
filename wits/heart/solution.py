# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:24:58 2021

Heart Disease Data - Machine Learning Classification

@author: john.atherfold
"""

#%% 0. Import the python libraries you think you'll require
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.svm import SVC
from sklearn import tree

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')

heartData = pd.read_csv('./data/heart_failure_clinical_records_dataset.csv')

#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)

plt.figure()
plt.title('DEATH_EVENT')
heartData['DEATH_EVENT'].value_counts().plot(kind='bar')

pd.plotting.scatter_matrix(heartData)

for columnName in heartData.columns[:-1]:
    plt.figure()
    heartData2 = heartData.groupby([columnName,"DEATH_EVENT"])[columnName].count().unstack('DEATH_EVENT').fillna(0)
    heartData2[[0,1]].plot(kind='bar', stacked=True)
    print(heartData2)

#   2.2 Exploring the dependence in the predictors/inputs

# This particular feature is logged - spreads the data out a bit more, as this was
# a highly tailed distribution
heartData['creatinine_phosphokinase'] = np.log(heartData['creatinine_phosphokinase'])

plt.figure()
sns.pairplot(heartData)


#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.

# Split the Test Set stratified on our target class. This can be checked by plotting
# histograms of yTest and yTrainValid and comparing them. They should look the same.
xTrainValid, xTest, yTrainValid, yTest = model_selection.train_test_split(
    heartData[heartData.columns[:-1]].values, heartData[heartData.columns[-1]].values,
    test_size = int(0.15*300), stratify=heartData[heartData.columns[-1]])

# Balance the data set by undersampling the major class (this can also be done
# by oversampling the minor class)

deathEventIndex = np.where(yTrainValid == 1)[0]
nonDeathEventIndex = np.where(yTrainValid == 0)[0]
underSamples = random.choices(list(nonDeathEventIndex),k=len(deathEventIndex))
xTrainValid = np.concatenate((xTrainValid[deathEventIndex],xTrainValid[underSamples]))
yTrainValid = np.concatenate((yTrainValid[deathEventIndex],yTrainValid[underSamples]))

#Check for balance
plt.hist(yTrainValid)

# Extract categorical predictors. We'd like to scale the data and run a PCA on
# it, but this should not be done on binary/categorical data

a = heartData[heartData.columns[:-1]].apply(np.unique)
logicalCategoricalColumns = np.array(a.apply(len) == 2)

#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

# There is a class imbalance in the data. Lets assume all data points belong to
# the same class as a start, and compare our models against that.

print(metrics.confusion_matrix(heartData['DEATH_EVENT'], np.zeros([len(heartData),1])))
print(metrics.f1_score(heartData['DEATH_EVENT'], np.zeros([len(heartData),1])))
print(metrics.accuracy_score(heartData['DEATH_EVENT'], np.zeros([len(heartData),1])))

# The model has an accuracy of 67.9%, which may seem good, but it's actually just a random guess.
# Also it's noteworthy that the F1 score is 0. A good model has an F1 score close to 1

#%% 5. Identify a Suitable Machine Learning Model

crossValObj = KFold(n_splits=20)

yValidFull = []
yHatFull = []

for trainIdx, validIdx in crossValObj.split(xTrainValid):
    xTrain, xValid = xTrainValid[trainIdx], xTrainValid[validIdx]
    yTrain, yValid = yTrainValid[trainIdx], yTrainValid[validIdx]

    # Remove the categorical data
    xTrainCat = xTrain[:,logicalCategoricalColumns]
    xTrainCts = xTrain[:,~logicalCategoricalColumns]
    xValidCat = xValid[:,logicalCategoricalColumns]
    xValidCts = xValid[:,~logicalCategoricalColumns]

    # Create the scaler object to scale the data. Fit scaler on Training data and
    # Apply scaler on both Training and Validation data. Prevent data leakage.

    scaler = preprocessing.StandardScaler().fit(xTrainCts)
    xTrainScaled = scaler.transform(xTrainCts)
    xValidScaled = scaler.transform(xValidCts)

    # Create principal component from training data. Fit PCA on training data only,
    # and project validation data onto those PCs
    pca = PCA()
    pca.fit(xTrainScaled)
    xTrainPCs = pca.transform(xTrainScaled)
    xValidPCs = pca.transform(xValidScaled)

    # Append categorical data back to continuous data
    xTrain = np.concatenate((xTrainPCs,xTrainCat), axis=1)
    xValid = np.concatenate((xValidPCs,xValidCat), axis=1)

    #sns.pairplot(pd.DataFrame(np.concatenate((xTrain, np.transpose(np.expand_dims(yTrain, axis=0))), axis = 1)))

    # Fit the model and test accuracy - UNCOMMENT EACH SET OF LINES AT A TIME
    # TO VIEW RESPECTIVE MODEL PERFORMANCES

    # Logistic Regression - Not bad. 75% accuracy and 0.745 F1 score
    # mdl = linear_model.LogisticRegression()

    # k-NN -  You could check various values for k, and choose the best one
    # based on the total k-fold error (F1 score). This is hyper-parameter
    # optimisation.
    # mdl = KNeighborsClassifier(n_neighbors = 5)

    # SVM - Best model according to k-fold cross-validation results
    mdl = SVC(C=2, kernel = 'rbf', degree=10)

    # Decision Tree - Needs hyperparameter tuning, but most results aren't bad
    # mdl = tree.DecisionTreeClassifier()


    mdl.fit(xTrain, yTrain)
    yHatTrain = mdl.predict(xTrain)
    # print(metrics.confusion_matrix(yTrain, yHatTrain))
    yHat = mdl.predict(xValid)
    yValidFull = np.concatenate((yValidFull,yValid))
    yHatFull = np.concatenate((yHatFull,yHat))

print(metrics.confusion_matrix(yValidFull, yHatFull))
print(metrics.f1_score(yValidFull, yHatFull))
print(metrics.accuracy_score(yValidFull, yHatFull))


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
xTrainValidCat = xTrainValid[:,logicalCategoricalColumns]
xTrainValidCts = xTrainValid[:,~logicalCategoricalColumns]
xTestCat = xTest[:,logicalCategoricalColumns]
xTestCts = xTest[:,~logicalCategoricalColumns]

# Create the scaler object to scale the data. Fit scaler on Training data and
# Apply scaler on both Training and Validation data. Prevent data leakage.

scaler = preprocessing.StandardScaler().fit(xTrainValidCts)
xTrainValidScaled = scaler.transform(xTrainValidCts)
xTestScaled = scaler.transform(xTestCts)

# Create principal component from training data. Fit PCA on training data only,
# and project validation data onto those PCs
pca = PCA()
pca.fit(xTrainValidScaled)
xTrainValidPCs = pca.transform(xTrainValidScaled)
xTestPCs = pca.transform(xTestScaled)

# Append categorical data back to continuous data
xTrainValid = np.concatenate((xTrainValidPCs,xTrainValidCat), axis=1)
xTest = np.concatenate((xTestPCs,xTestCat), axis=1)

mdl = SVC(C=2, kernel = 'rbf', degree=10)
mdl.fit(xTrainValid, yTrainValid)
yHatTest = mdl.predict(xTest)

print(metrics.confusion_matrix(yTest, yHatTest))
print(metrics.f1_score(yTest, yHatTest))
print(metrics.accuracy_score(yTest, yHatTest))

print(metrics.confusion_matrix(yTest, np.zeros([len(yTest),1])))
print(metrics.f1_score(yTest, np.zeros([len(yTest),1])))
print(metrics.accuracy_score(yTest, np.zeros([len(yTest),1])))

#%% 7. Add Complexity if Required
# In this case, the complexity is added in the functionality, and is iterated
# upon by uncommenting the appropriate model-fitting lines

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question
