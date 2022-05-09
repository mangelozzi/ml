# -*- coding: utf-8 -*-
"""
Concrete Example

@author: Michael Angelozzi
"""

#%% 0. Import the python libraries you think you'll require

from matplotlib import pyplot as plt
from sklearn import tree, metrics, linear_model, preprocessing, decomposition, model_selection, pipeline
from skopt import BayesSearchCV
import numpy as np
import pandas as pd
import seaborn as sb

PLOT = True

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')
df = pd.read_csv('concrete_data.csv')
# print(df)

#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)
# PLOT and sb.pairplot(df)

#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.


#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

predictors = df[df.columns[:-1]]
responses = df['Strength']

n = int(0.85 * len(responses))
x_train_valid = predictors[:n]
y_train_valid = responses[:n]
x_test = predictors[n:]
y_test = responses[n:]

model = linear_model.LinearRegression()
model.fit(x_train_valid, y_train_valid)

y_pred_train_valid = model.predict(x_train_valid)
y_pred_test = model.predict(x_test)


# plt.figure()
# plt.hist(y_train_valid - y_pred_train_valid)
# plt.title("Training/Validation Residuals")
mse_train_valid = np.mean((y_train_valid - y_pred_train_valid)**2)  # mean squared errors
print(f"\nMSE (mean squared error) training/validation: {round(mse_train_valid, 2)}")

# plt.figure()
# plt.hist(y_pred_test - y_test)
# plt.title("Test Residuals")
mse_test = np.mean((y_test - y_pred_test)**2)  # mean squared errors
print(f"\nMSE (mean squared error) test: {round(mse_test, 2)}")

# plt.figure()
# ax = y_train_valid.plot()
# plt.plot(y_pred_train_valid)
# plt.title('Training-validation predictions with responses')
#
# plt.figure()
# ax = y_test.plot(use_index=False)
# plt.plot(y_pred_test)
# plt.title('Test predictions with responses')

train_valid_results = pd.DataFrame()
train_valid_results['Actual Value'] = y_train_valid
train_valid_results['Baseline'] = y_pred_train_valid
train_valid_results['Baseline Error'] = train_valid_results['Actual Value'] - train_valid_results['Baseline']

test_results = pd.DataFrame()
test_results['Actual Value'] = y_test
test_results['Baseline'] = y_pred_test
test_results['Baseline Error'] = test_results['Actual Value'] - test_results['Baseline']


#%% 5. Identify a Suitable Machine Learning Model

cvo = model_selection.KFold(n_splits = 10)  # cross validation object

pipe_params = [
    ('scaler', preprocessing.StandardScaler()),
    ('poly', preprocessing.PolynomialFeatures()),
    ('pca', decomposition.PCA()),
    ('regression', linear_model.Lasso()),
]
pipe = pipeline.Pipeline(pipe_params)
params = {
    'poly__degree': np.arange(1,8),  # vary 'degree' param of 'poly'
    'regression__alpha': np.logspace(-9, 1),  # vary 'alpha' param of 'regression'
}
# opt = optimizer object
# Will split according to cross validation object, apply the pipe line to the data
opt = BayesSearchCV(pipe, params, cv=cvo, verbose=5, n_iter=50) ## 10 TODO
search_results = opt.fit(x_train_valid, y_train_valid)
print(search_results.best_params_) # criterion: entropy, min_samples_leaf: 5
# Example
# OrderedDict([('poly__degree', 2), ('regression__alpha', 1.0481131341546852e-08)])

#%% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared

# Set Optimal Parameters Found in SearchResults
pipe.set_params(
    poly__degree = search_results.best_params_.get('poly__degree'),
    regression__alpha = search_results.best_params_.get('regression__alpha'),
)

# Fit pipe to train/validation data
pipe.fit(x_train_valid, y_train_valid)
y_pred_train_valid = pipe.predict(x_train_valid)
train_valid_results['Poly Model'] = y_pred_train_valid
train_valid_results['Poly Model Error'] = train_valid_results['Actual Value'] - train_valid_results['Poly Model']

# Fit pipe to test data
pipe.fit(x_test, y_test)
y_pred_test = pipe.predict(x_test)
test_results['Poly Model'] = y_pred_test
test_results['Poly Model Error'] = test_results['Actual Value'] - test_results['Poly Model']

# Get Relevant Plots
rmseTrain = np.mean(train_valid_results['Poly Model Error']**2)**0.5
#
plt.figure()
ax = train_valid_results['Baseline Error'].plot(kind = 'hist', alpha = 0.5, bins = 20)
train_valid_results['Poly Model Error'].plot(kind = 'hist', alpha = 0.5, bins = 20, ax = ax)
plt.legend(('Baseline', 'Poly Model'))
plt.title('Residuals')

rmseTest = np.mean(test_results['Poly Model Error']**2)**0.5

plt.figure()
ax = test_results['Baseline Error'].plot(kind = 'hist', alpha = 0.5, bins = 12)
test_results['Poly Model Error'].plot(kind = 'hist', alpha = 0.5, bins = 12, ax = ax)
plt.legend(('Baseline', 'Poly Model'))
plt.title('Test Results')

plt.figure()
ax = train_valid_results['Actual Value'].plot()
train_valid_results['Baseline'].plot(ax = ax)
train_valid_results['Poly Model'].plot(ax = ax)
plt.legend(('Actual Value','Baseline Model','Poly Model'))
plt.title('Train-Valid Data')

plt.figure()
ax = test_results['Actual Value'].plot()
test_results['Baseline'].plot(ax = ax)
test_results['Poly Model'].plot(ax = ax)
plt.legend(('Actual Value','Baseline Model','Poly Model'))
plt.title('Test Data')

#
# print('Baseline RMSE Training: ', rmseTrainBaseline)
# print('Baseline RMSE Testing: ', rmseTestBaseline)
#
# print('Baseline R Squared Training: ',
#       r2_score(yTrainValidResults['Actual Value'], yTrainValidResults['Baseline']))
# print('Baseline R Squared Testing: ',
#       r2_score(test_results['Actual Value'], test_results['Baseline']))
#
#
# print('Poly Model RMSE Training: ', rmseTrain)
# print('Poly Model RMSE Testing: ', rmseTest)
#
# print('Poly Model R Squared Training: ',
#       r2_score(yTrainValidResults['Actual Value'], yTrainValidResults['Poly Model']))
# print('Poly Model R Squared Testing: ',
#       r2_score(test_results['Actual Value'], test_results['Poly Model']))


#%% 7. Add Complexity if Required

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question


# Show the plots in none interactive mode
if PLOT:
    plt.show(block=False)
    input('Press enter to exit')
    plt.close('all')

