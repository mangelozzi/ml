# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:24:58 2021

Mushroom Data - Machine Learning Classification

@author: john.atherfold
"""

#%% 0. Import the python libraries you think you'll require
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import KFold

# Disable more than 20 plots warning
plt.rcParams.update({'figure.max_open_warning': 0})

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')
df = pd.read_csv('mushrooms.csv')
# print(df.head(3))
# print(df.columns[1:])

PLOT = False

for column in df.columns[1:]:
    series = df.groupby([column, "class"])  # get just the column and class
    data = series[column]  # access the data
    counts = data.count()
    unstacked = counts.unstack()
    valid = unstacked.fillna(0)  # replace NaN with zeros instead
    # Example output:
    # class       e       p
    # odor
    # a       400.0     0.0
    # c         0.0   192.0
    # f         0.0  2160.0
    # l       400.0     0.0
    # m         0.0    36.0
    # n      3408.0   120.0
    # p         0.0   256.0
    # s         0.0   576.0
    # y         0.0   576.0
    PLOT and valid.plot(kind="bar", stacked="True")
    print(valid)
# Show the plots in none interactive mode
if PLOT:
    plt.show(block=False)
    input('Press enter to exit')
    plt.close('all')

#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)


#   2.2 Exploring the dependence in the predictors/inputs
#           In this case, we have categorical predictors, so we have to consider
#           the dependence of the frequencies of sampling two random variables
#
#           Null Hypothesis: There are no non-random associations between variables (the variables are independent)
#           Alternative Hypothesis: There are non-random associations between variables (the variables are dependent)
#           If significance level (p-value) is very small (less than 0.05), the Null hypothesis is rejected

# This exercise is shown for two pairs of variables. Ideally it should be done for
# all pairs of variables.
cross_tab = pd.crosstab(df['cap-shape'], df['cap-surface'])
pval = stats.chi2_contingency(cross_tab.values)[1]
# If the variables are dependant we can drop one of the values
print(f"pval is: {pval}, less than < 0.05: {pval < 0.05}, conclusion: {'the variables are dependent' if pval < 0.05 else 'variables are independant'}")

#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.
# Convert categorical data to binary feature columns
dummy = pd.get_dummies(df[df.columns[1:]])
training_len = round(len(df) * 0.85)
x_train = dummy[:training_len]
y_train = df['class'][:training_len]
x_test = dummy[training_len:]
y_test = df['class'][training_len:]

#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)
y_hat_baseline = [True] * len(y_train)
# CONFUSION MATRIX:
# Inputs
# TRUE     TP  |  FP
# FALSE    FN  |  TN
#         TRUE | FALSE
#           Outputs
dummy_confusion_matrix = metrics.confusion_matrix(y_train == 'e', y_hat_baseline)
dummy_accuracy = metrics.accuracy_score(y_train == 'e', y_hat_baseline)
print(f"\nDUMMY GUESS: Always guess edible")
print(f"Confusion matrix: {dummy_confusion_matrix}")
print(f"The accuracy is {round(dummy_accuracy*100)}%")

#%% 5. Identify a Suitable Machine Learning Model
cvo = KFold(n_splits=10)  # cross validation object
y_hat_valid_total = []

# we split our training data 10 times into training and validation 10 times:
# data = 85% of data using for training (x_train)
# 1. Take the first 10% of data use that as model validation data
# 2. Use the remaining data as training data
# 3. We then take the next 10% of data, and repeat the steps 1, 2, until we go through all the data
for i_train, i_valid in cvo.split(x_train):
    x_train_split = x_train.loc[i_train]
    x_valid_split = x_train.loc[i_valid]
    y_train_split = y_train.loc[i_train]
    y_valid_split = y_train.loc[i_valid]
    model = tree.DecisionTreeClassifier()
    model.fit(x_train_split, y_train_split == 'p')
    y_hat_valid = model.predict(x_valid_split)
    # np.concatenate((y_hat_valid_total, y_hat_valid), axis=0)
    y_hat_valid_total.extend(y_hat_valid)

confusion_matrix = metrics.confusion_matrix(y_train == 'p', y_hat_valid_total)
accuracy = metrics.accuracy_score(y_train == 'p', y_hat_valid_total)
print(f"\nMODEL")
print(f"Confusion matrix: {confusion_matrix}")
print(f"The accuracy is {round(accuracy*100, 2)}%")

#%% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train == 'p')
y_hat_test = model.predict(x_test)
test_confusion_matrix = metrics.confusion_matrix(y_test == 'p', y_hat_test)
test_accuracy = metrics.accuracy_score(y_test == 'p', y_hat_test)
print(f"\nTEST")
print(f"Confusion matrix: {test_confusion_matrix}")
print(f"The accuracy is {round(test_accuracy*100, 2)}%")

#%% 7. Add Complexity if Required

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question
