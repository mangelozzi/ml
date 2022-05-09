# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:08:48 2021

Concrete Example

@author: john.atherfold
"""

#%% 0. Import the python libraries you think you'll require

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')

concreteData = pd.read_csv('./data/concrete_data.csv')
concreteData = concreteData.sample(frac = 1).reset_index(drop = True) # Shuffles the data around (I have a feeling that it's ordered in some way, and we don't necessarily want that)

#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)

sns.pairplot(concreteData)

#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.

predictors = concreteData[concreteData.columns[:-1]]
responses = concreteData[concreteData.columns[-1]]

xTrainValid = predictors[0:int(0.85*len(predictors))]
yTrainValid = responses[0:int(0.85*len(predictors))]

xTest = predictors[int(0.85*len(predictors)):]
yTest = responses[int(0.85*len(predictors)):]

#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

mdl = LinearRegression()
mdl.fit(xTrainValid, yTrainValid)

yHatTrainValidBaseline = mdl.predict(xTrainValid)
yHatTestBaseline = mdl.predict(xTest)

yTrainValidResults = pd.DataFrame()
yTrainValidResults['Actual Value'] = yTrainValid
yTrainValidResults['Baseline'] = yHatTrainValidBaseline
yTrainValidResults['Baseline Error'] = yTrainValidResults['Actual Value'] - yTrainValidResults['Baseline']

yTestResults = pd.DataFrame()
yTestResults['Actual Value'] = yTest
yTestResults['Baseline'] = yHatTestBaseline
yTestResults['Baseline Error'] = yTestResults['Actual Value'] - yTestResults['Baseline']

rmseTrainBaseline = np.mean(yTrainValidResults['Baseline Error']**2)**0.5
plt.figure()
plt.hist(yTrainValidResults['Baseline Error'])
plt.title('Baseline - Residuals')

rmseTestBaseline = np.mean(yTestResults['Baseline Error']**2)**0.5
plt.figure()
plt.hist(yTestResults['Baseline Error'])
plt.title('Baseline - Test Errors')

plt.figure()
ax = yTrainValidResults['Actual Value'].plot()
yTrainValidResults['Baseline'].plot(ax = ax)
plt.title('Baseline - Train-Valid Data')

plt.figure()
ax = yTestResults['Actual Value'].plot()
yTestResults['Baseline'].plot(ax = ax)
plt.title('Baseline - Test Data')

#%% 5. Identify a Suitable Machine Learning Model

crossValObj = KFold(n_splits = 10)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('poly', PolynomialFeatures()),
                 ('pca', PCA()),
                 ('regression', Lasso())])

params = {
    'poly__degree': np.arange(1,8,1),
    'regression__alpha': np.logspace(-9, 1, 1000)
    }

opt = BayesSearchCV(pipe, params, cv = crossValObj, verbose = 5, n_iter = 100,
                    n_jobs = -1)
searchResults = opt.fit(xTrainValid, yTrainValid)
# Cross-validation has been completed
print('Cross Validation Completed')
print('--------------------------')
print('Best Polynomial Degree: ' + str(searchResults.best_params_.get('poly__degree')))
print('Best Regression Alpha: ' + str(searchResults.best_params_.get('regression__alpha')))
crossValidationScore = cross_val_score(pipe, xTrainValid, yTrainValid, cv = crossValObj)
print('Cross Validation Score: ' + str(np.mean(crossValidationScore)))
print('--------------------------')

#%% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared

# Set Optimal Parameters Found in SearchResults
pipe.set_params(poly__degree = searchResults.best_params_.get('poly__degree'),
                regression__alpha = searchResults.best_params_.get('regression__alpha'))

# Fit whole pipeline workflow to all Train and Valid Data
pipe.fit(xTrainValid, yTrainValid)

# Get Training Results
yHatTrainValid = pipe.predict(xTrainValid)
yTrainValidResults['Poly Model'] = yHatTrainValid
yTrainValidResults['Poly Model Error'] = yTrainValidResults['Actual Value'] - yTrainValidResults['Poly Model']

# Get Testing Results
yHatTest = pipe.predict(xTest)
yTestResults['Poly Model'] = yHatTest
yTestResults['Poly Model Error'] = yTestResults['Actual Value'] - yTestResults['Poly Model']

# Get Relevant Plots

rmseTrain = np.mean(yTrainValidResults['Poly Model Error']**2)**0.5

plt.figure()
ax = yTrainValidResults['Baseline Error'].plot(kind = 'hist', alpha = 0.5, bins = 20)
yTrainValidResults['Poly Model Error'].plot(kind = 'hist', alpha = 0.5, bins = 20, ax = ax)
plt.legend(('Baseline', 'Poly Model'))
plt.title('Residuals')

rmseTest = np.mean(yTestResults['Poly Model Error']**2)**0.5

plt.figure()
ax = yTestResults['Baseline Error'].plot(kind = 'hist', alpha = 0.5, bins = 12)
yTestResults['Poly Model Error'].plot(kind = 'hist', alpha = 0.5, bins = 12, ax = ax)
plt.legend(('Baseline', 'Poly Model'))
plt.title('Test Results')

plt.figure()
ax = yTrainValidResults['Actual Value'].plot()
yTrainValidResults['Baseline'].plot(ax = ax)
yTrainValidResults['Poly Model'].plot(ax = ax)
plt.legend(('Actual Value','Baseline Model','Poly Model'))
plt.title('Train-Valid Data')

plt.figure()
ax = yTestResults['Actual Value'].plot()
yTestResults['Baseline'].plot(ax = ax)
yTestResults['Poly Model'].plot(ax = ax)
plt.legend(('Actual Value','Baseline Model','Poly Model'))
plt.title('Test Data')

print('Baseline RMSE Training: ', rmseTrainBaseline)
print('Baseline RMSE Testing: ', rmseTestBaseline)

print('Baseline R Squared Training: ',
      r2_score(yTrainValidResults['Actual Value'], yTrainValidResults['Baseline']))
print('Baseline R Squared Testing: ',
      r2_score(yTestResults['Actual Value'], yTestResults['Baseline']))


print('Poly Model RMSE Training: ', rmseTrain)
print('Poly Model RMSE Testing: ', rmseTest)

print('Poly Model R Squared Training: ',
      r2_score(yTrainValidResults['Actual Value'], yTrainValidResults['Poly Model']))
print('Poly Model R Squared Testing: ',
      r2_score(yTestResults['Actual Value'], yTestResults['Poly Model']))

# Much better result - Training and testing sets have consistent results, and
# our Poly Model performs much better than the baseline.

#%% 7. Add Complexity if Required

# I'm fairly pleased with these for now, but feel free to try a SVR, or KernelPCA
# with another linear model and a RBF Kernel, or something.

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question

# Get each predictor name and their corresponding coeficient value. Get a sense
# of feature importance.

linearMdl = pipe.named_steps['regression']
pca = pipe.named_steps['pca']

pcaComps = pca.components_
regressionCoefs = linearMdl.coef_
predictorWeights = np.matmul(regressionCoefs, pcaComps)

featureNames = pd.Series(pipe.named_steps['poly'].get_feature_names(concreteData.columns))
importantIndicators = np.flip(np.argsort(np.abs(predictorWeights)))
top20Columns = np.flip(featureNames[importantIndicators[0:20]])
top20Weights = np.flip(predictorWeights[importantIndicators[0:20]])

plt.figure()
plt.barh(top20Columns, top20Weights, align = 'center')
plt.title('Linear Model Top 20 Features')
