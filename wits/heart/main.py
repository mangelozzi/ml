# -*- coding: utf-8 -*-
"""
Heart Disease Data - Machine Learning Classification

@author: Michael Angelozzi
"""

#%% 0. Import the python libraries you think you'll require
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn import model_selection, metrics, preprocessing, decomposition, neighbors, linear_model

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# print(df)

#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)
PLOT = False

plt.figure()
plt.title('Death Event')
# .counts will give counts, value_counts groups values
plot_data = df['DEATH_EVENT'].value_counts()
# We see its like a 2/3 and 1/3 split between the labels
# print(plot_data)

# Show bar plot of the death event split
PLOT and plot_data.plot(kind="bar")

# Examine the heart data - Not very readible
# pd.plotting.scatter_matrix(df)
PLOT and  sb.pairplot(df)

for column in df.columns[:-1]:
    grouped_df = df.groupby([column, 'DEATH_EVENT'])[column].count().unstack().fillna(0)
    columns = grouped_df[[0,1]]  # get both columns
    PLOT and columns.plot(kind="bar", stacked=True)


# Show the plots in none interactive mode
if PLOT:
    plt.show(block=False)
    input('Press enter to exit')
    plt.close('all')


#   2.2 Exploring the dependence in the predictors/inputs

# In this case the data is "cross-sectional", meaning its not streaming, all the data for the person is just there (no time dependance), i.e. a full cross section

# This particular feature is logged - spreads the data out a bit more, as this was
# a highly tailed distribution

predictors = df[df.columns[:-1]]
response = df['DEATH_EVENT']
x_train_valid, x_test, y_train_valid, y_test = model_selection.train_test_split(predictors, response, train_size=0.85, stratify=response)

print('\nCheck the data is split with same ratio of results')
print(y_train_valid.value_counts())
print(y_test.value_counts())

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

# age                         [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47....
# anaemia                                                                [0, 1]
# creatinine_phosphokinase    [23, 30, 47, 52, 53, 54, 55, 56, 57, 58, 59, 6...
# diabetes                                                               [0, 1]
# ejection_fraction           [14, 15, 17, 20, 25, 30, 35, 38, 40, 45, 50, 5...
# high_blood_pressure                                                    [0, 1]
# platelets                   [25100.0, 47000.0, 51000.0, 62000.0, 70000.0, ...
# serum_creatinine            [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.18...
# serum_sodium                [113, 116, 121, 124, 125, 126, 127, 128, 129, ...
# sex                                                                    [0, 1]
# smoking                                                                [0, 1]
# time                        [4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 20, 2...
# Convert binary features, e.g. sex, diabetes to separate categories
unique_columns = predictors.apply(np.unique)
# Array of boolean values where True = a binary feature to be split out
logical_categorical_columns = np.array(unique_columns.apply(len) == 2)

#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

# There is a class imbalance in the data. Lets assume all data points belong to
# the same class as a start, and compare our models against that.

y_pred = np.zeros(len(y_train_valid))  # What lecturer calls YHatBaseline
confusion_matrix = metrics.confusion_matrix(y_train_valid, y_pred)
accuracy = metrics.accuracy_score(y_train_valid, y_pred)
f1_score = metrics.f1_score(y_train_valid, y_pred)
print("\nAlways True Confusion matrix:", confusion_matrix)
print(f"Accuracy: {round(accuracy * 100)}%, F1 Score: {f1_score}")

# The model has an accuracy of 67.9%, which may seem good, but it's actually just a random guess.
# Also it's noteworthy that the F1 score is 0. A good model has an F1 score close to 1

#%% 5. Identify a Suitable Machine Learning Model
# y_pred = [0] * len(y_train_valid)

# 5 splits good default, 10 better, 20 very good is possible
# cvo = Cross validation object
cvo = model_selection.KFold(n_splits=20)  # cross validation object

y_valid = []
y_pred = []
for i_train, i_valid in cvo.split(x_train_valid):
    x_train_split = x_train_valid.iloc[i_train]
    x_valid_split = x_train_valid.iloc[i_valid]
    y_train_split = y_train_valid.iloc[i_train]
    y_valid_split = y_train_valid.iloc[i_valid]
    # Remove categoricals
    x_train_split_cats  = x_train_split.iloc[:, logical_categorical_columns]  # categoricals
    x_train_split_conts = x_train_split.iloc[:, ~logical_categorical_columns]  # continuous
    x_valid_split_cats  = x_valid_split.iloc[:, logical_categorical_columns]  # categoricals
    x_valid_split_conts = x_valid_split.iloc[:, ~logical_categorical_columns]  # continuous

    # We find the mean and std deviation using only our training data, then apply the same z-score transformation to our validation data
    scaler = preprocessing.StandardScaler().fit(x_train_split_conts)
    x_train_scaled = scaler.transform(x_train_split_conts)
    x_valid_scaled = scaler.transform(x_valid_split_conts)

    # Principal component analysis, reduce dimensionality, pull out co-linearity (features that are linearly dependant)
    # We find the PCA components only our training data, then reduce both the training and validation data using those ocmponents (treat the validation data as unknowns at this point in time like training data)
    pca = decomposition.PCA().fit(x_train_scaled)
    x_train_pcs = pca.transform(x_train_scaled)
    x_valid_pcs = pca.transform(x_valid_scaled)

    x_train_clean = np.concatenate([x_train_pcs, x_train_split_cats], axis=1)
    x_valid_clean = np.concatenate([x_valid_pcs, x_valid_split_cats], axis=1)

    # model = neighbors.KNeighborsClassifier  # lecturer changed his mind
    model = linear_model.LogisticRegression()
    model.fit(x_train_clean, y_train_split)

    # y_pred_train = model.predict(x_train)
    y_pred_valid = model.predict(x_valid_clean)

    y_valid.extend(y_valid_split)
    y_pred.extend(y_pred_valid)

confusion_matrix = metrics.confusion_matrix(y_train_valid, y_pred)
accuracy = metrics.accuracy_score(y_train_valid, y_pred)
f1_score = metrics.f1_score(y_train_valid, y_pred)
print("Validation Model Results")
print("\nConfusion matrix:", confusion_matrix)
print(f"Accuracy: {round(accuracy * 100)}%, F1 Score: {f1_score}")

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
