# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:24:58 2021

Mushroom Data - Machine Learning Classification

@author: john.atherfold
"""

#%% 0. Import the python libraries you think you'll require
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np
from skopt import BayesSearchCV

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')

mushroomData  = pd.read_csv('./data/mushrooms.csv')

#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)

for column in mushroomData.columns[1:]:
    mushroomSeries = mushroomData.groupby([column,"class"])[column].count().unstack("class").fillna(0)
    mushroomSeries[["e","p"]].plot(kind = "bar", stacked = True)

#   2.2 Exploring the dependence in the predictors/inputs
#           In this case, we have categorical predictors, so we have to consider
#           the dependence of the frequencies of sampling two random variables
#
#           Null Hypothesis: There are no non-random associations between variables (the variables are independent)
#           Alternative Hypothesis: There are non-random associations between variables (the variables are dependent)
#           If significance level (p-value) is very small (less than 0.05), the Null hypothesis is rejected

# This exercise is shown for two pairs of variables. Ideally it should be done for
# all pairs of variables.

crossTab = pd.crosstab(mushroomData['cap-shape'], mushroomData['cap-surface'], margins = True)
print(crossTab)
chi2, pval = stats.chi2_contingency(crossTab.values)[0:2]


#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.

dummyMushroomData = pd.get_dummies(mushroomData[mushroomData.columns[1:]])

xTrainValidData = dummyMushroomData[0:round(0.85*len(mushroomData))]
yTrainValidData = mushroomData['class'][0:round(0.85*len(mushroomData))]

xTestData = dummyMushroomData[round(0.85*len(mushroomData)):]
yTestData = mushroomData['class'][round(0.85*len(mushroomData)):]

#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

yHatBaseline = [True for i in range(len(yTrainValidData))]
print(metrics.confusion_matrix(yTrainValidData == 'e', yHatBaseline))
print(metrics.accuracy_score(yTrainValidData == 'e', yHatBaseline))

#%% 5. Identify a Suitable Machine Learning Model

crossValObj = KFold(n_splits = 10)

yHatValidTotal = []
for trainIdx, validIdx in crossValObj.split(xTrainValidData):
    xTrain, xValid = xTrainValidData.loc[trainIdx], xTrainValidData.loc[validIdx]
    yTrain, yValid = yTrainValidData.loc[trainIdx], yTrainValidData.loc[validIdx]
    mdl = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 15)
    mdl.fit(xTrain, yTrain == 'p')
    yHatValid = mdl.predict(xValid)
    yHatValidTotal = np.concatenate((yHatValidTotal, yHatValid), axis = 0)

print(metrics.confusion_matrix(yTrainValidData == 'p', yHatValidTotal))
print(metrics.accuracy_score(yTrainValidData == 'p', yHatValidTotal))

#%% Looking for Optimal Hyperparameters

crossValObj = KFold(n_splits = 10)

mdl = tree.DecisionTreeClassifier()
params = {'criterion': ['gini','entropy'],
          'min_samples_leaf': np.arange(1,31,1)}
opt = BayesSearchCV(mdl, params, cv = crossValObj, verbose = 5, n_iter = 10)
searchResults = opt.fit(xTrainValidData, yTrainValidData)

#%% Test the Model

mdl = tree.DecisionTreeClassifier(criterion = searchResults.best_params_.get('criterion'),
                                  min_samples_leaf = searchResults.best_params_.get('min_samples_leaf'))
mdl.fit(xTrainValidData, yTrainValidData == 'p')
yHatTest = mdl.predict(xTestData)

print(metrics.confusion_matrix(yTestData == 'p', yHatTest))
print(metrics.accuracy_score(yTestData == 'p', yHatTest))

#%% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared

#%% 7. Add Complexity if Required

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question
