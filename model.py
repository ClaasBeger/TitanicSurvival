import pandas as pd

import torch

import numpy as np

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

#%%

# Check for categorical variables

train_df.applymap(np.isreal).all(0)


#%%

# Encode more diverse categorical variable Embarked using OneHotEncoder

# drop the two missing values for Embarked (conforms to the <5% Rule of Thumb)
train_df.dropna(subset=['Embarked'], inplace=True)

# =============================================================================
# from sklearn.preprocessing import OneHotEncoder
# 
# column = 'Embarked'
# 
# encoder = OneHotEncoder(sparse=False)
# 
# encoded_column = encoder.fit_transform(train_df[[column]])
# 
# encoded_column_names = encoder.get_feature_names([column])
# 
# encoded_df = pd.DataFrame(encoded_column, columns=encoded_column_names)
# 
# train_df = pd.concat([train_df, encoded_df], axis=1)
# 
# train_df = train_df.drop(labels=column, axis=1)
# =============================================================================

#%%

# All features are now numerical. Check for null values

print(train_df.info())

# Null values present in age

# To fill the missing values for age, I will first compute the mean for each of the genders
female_avg = train_df.groupby("Sex").Age.mean()["female"]
male_avg = train_df.groupby("Sex").Age.mean()["male"]

# Now I assign the missing values to the mean, based on the gender of the passenger
mask = (train_df['Sex'] == 'male') & train_df['Age'].isnull()
maskf = (train_df['Sex'] == 'female') & train_df['Age'].isnull()
train_df.loc[mask, 'Age'] = male_avg
train_df.loc[maskf, 'Age'] = female_avg

#%%

# Cabin has many missing values, it would be useless to assign some average or median cabin value to it (for example a medium 
# location/quality, because the most useful information is whether 
# the passenger actually has a cabin or not. I will omit the cabin number, and instead just keep the Letter, which represents
# the general location/quality of the cabin. For the missing Values, I will insert "N"
maskLetter = train_df['Cabin'].isnull()
train_df.loc[maskLetter, 'Cabin'] = 'N'

#While we risk losing information by deleting the exact cabin number, this way the feature will be far more meaningful for 
# a Decision Tree Model
train_df.loc[train_df['Cabin'].notnull(),'Cabin'] = train_df['Cabin'].str[0]

#%%

# Check for missing values in the testing data Set
test_df.info()
# Missing Values in Age, Fare, and Cabin

# Similarly to the training data, we fill age up with the corresponding mean (computed on training set), We drop the missing 
# row in Fare (same reason as for Embarked) and we fill up the values with 'N' in cabin, respectively replace the value by
# the prefix
mask = (test_df['Sex'] == 'male') & test_df['Age'].isnull()
maskf = (test_df['Sex'] == 'female') & test_df['Age'].isnull()
test_df.loc[mask, 'Age'] = male_avg
test_df.loc[maskf, 'Age'] = female_avg

# drop the one missing value for Fare (conforms to the <5% Rule of Thumb)
#test_df.dropna(subset=['Fare'], inplace=True)
# This leads to a Kaggle Submission Error, so I will instead replace the value with the average from train
test_df['Fare'].fillna(train_df['Fare'].mean(),inplace = True)

maskLetter = test_df['Cabin'].isnull()
test_df.loc[maskLetter, 'Cabin'] = 'N'

test_df.loc[test_df['Cabin'].notnull(),'Cabin'] = test_df['Cabin'].str[0]

#%%

from sklearn import preprocessing

# create a scaler using Z score normalization / Standardization on the Fare attribute
# The Fare varies greatly, thus we want to normalize this attribute
zscore_scaler = preprocessing.StandardScaler().fit(train_df[['Fare']])

#Apply the Zscore scaler Fare attribute and assign to a new Zscore column
train_df['Fare_zscore']=zscore_scaler.transform(train_df[['Fare']])

train_df['Fare_zscore'].describe()

#%%

# get dummy variables for categorical varialbes
sexdummy = pd.get_dummies(train_df['Sex'], drop_first=True)
embarkeddummy = pd.get_dummies(train_df['Embarked'],drop_first=True,prefix='Embarked')
cabindummy = pd.get_dummies(train_df['Cabin'],drop_first=True,prefix='Cabin')

#%%

# adding dummy columns to the existing dataframe
trainwithdummy = pd.concat([train_df,sexdummy,embarkeddummy,cabindummy],axis=1,sort=True)
trainwithdummy.info()

#%%

# drop columns that are not needed or have already been encoded
train_df=trainwithdummy.drop(columns=['Name','Sex','Ticket','Fare','Embarked'])

#%%

# view the updated dataframe
train_df.head()

#%%

# Apply same discretization and one-hot encoding to the testing set

# Make sure to apply to 
test_df['Fare_zscore']=zscore_scaler.transform(test_df[['Fare']])

# get dummy variables for categorical varialbes
sexdummy = pd.get_dummies(test_df['Sex'], drop_first=True)
embarkeddummy = pd.get_dummies(test_df['Embarked'],drop_first=True,prefix='Embarked')
cabindummy = pd.get_dummies(test_df['Cabin'],drop_first=True,prefix='Cabin')

testwithdummy = pd.concat([test_df,sexdummy,embarkeddummy,cabindummy],axis=1,sort=True)

# We have to add an empty column for Cabin_T, because there is no given passenger with it in the Test Set
testwithdummy['Cabin_T'] = 0
# drop columns that are not needed
test_df=testwithdummy.drop(columns=['Name','Sex','Ticket','Fare','Embarked'])
test_df.head()

#%%

# Save cleaned Train and cleaned Testing Data to csv


train_df.to_csv('titanic_cleaned')
test_df.to_csv('titanictest_cleaned')

#%%

# Define Target and Features

# Start building the Decision Tree

# define independent variables / attirbutes / features
features = ['Pclass','Age','SibSp','Parch','Fare_zscore','male','Embarked_Q','Embarked_S','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_N','Cabin_T']
# define one single target variable / label
target = ['Survived']

test_df[features].head()

#%%

# get defined training dataset
X = train_df[features]
y = train_df[target]

#%%

# import train split function
from sklearn.model_selection import train_test_split
# import libraries for cross validation
# do the split automatically multiple times via cross validation
# evenly distribute them into 5 folds (every fold holds more or less equal amount)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# import evaluation tools
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report # precision and recall
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# split train data into train and test, 60% in training and 40% in testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 777)

#%%

# define our model by using the default value

# specify criterion = 'entropy'
# specify max_depth (try out)
# min_samples_split (try out (probably something bigger than two, minimum))
# Reminder : min_samples_split specifies the minimum number of samples required to split an internal node,
# while min_samples_leaf specifies the minimum number of samples required to be at a leaf node.
# min_impurity_decrease (try out) (probably something bigger than .10)
# criterion='entropy', min_samples_split=10, min_samples_leaf=10
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

#%%

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize = (100,150))
tree.plot_tree(model,ax=None, fontsize=50)
plt.show()

#%%

# Accuracies from cross validation
#cv denotes number of folds
# output denotes accuracies on the val fold (take average for better representation)
print(cross_val_score(model, X, y, cv=10))

#%%

cross_val_score(model, X, y, cv=10).mean()
# we choose none of these models, but rather one model which trains on all the data (when we are satisfied with validation scores)

#%%

pred_label_cv = cross_val_predict(model, X, y, cv=10)
pred_label_cv

#%%

print("Confusion Matrix", '\n', confusion_matrix(y, pred_label_cv))
print("Classification Report:", '\n', classification_report(y, pred_label_cv))

#%%

# train a decision tree to train a model based on whole training set
# (Suppose we are content with validation result)
model_w = DecisionTreeClassifier()
model_w.fit(X,y)

#%%

# test evaluation result for 60% train data
pred_val_train = model_w.predict(X)

print("Accuracy from whole train data:",accuracy_score(y, pred_val_train, normalize=True, sample_weight=None))
print("Confusion Matrix", '\n', confusion_matrix(y, pred_val_train))
print("Classification Report:", '\n', classification_report(y, pred_val_train))

#%%

# create an array that holds the max_depth
# tree is too big, we need to fine-tune some variables
max_depth = np. linspace(1,32,32, endpoint=True) #<- try out different depth from 1 to 32
# but be careful not too underfit (depth too low-> learning is not enough)
# this is pre-tuning (only grow tree to certain depth) vs. post-tuning (cut off after growing big tree)

#%%

# create a loop to try out different depth value
# test accuracies in the train and test results and build a graph from it
train_results =[]
test_results = []
# create a loop to try out all the number from 1 to 32 for max_depths in a decision tree
for max_depth_i in max_depth:
    dt = DecisionTreeClassifier(max_depth=max_depth_i)
    dt.fit(X_train, y_train)
    
    train_pred = dt.predict(X_train)
    accuracy = accuracy_score(y_train, train_pred, normalize=True, sample_weight=None)
    train_results.append(accuracy)
    
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    test_results.append(accuracy)
    
#%%

from matplotlib.legend_handler import HandlerLine2D
# trend is decreasing for test accuracy (increasing for train)
# there is an optimum (3 or 4), which will help us fine tune the model
line1, = plt.plot(max_depth, train_results, 'b', label='Train accuracy')
line2, = plt.plot(max_depth, test_results, 'r', label='Test accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy score')
plt.xlabel('Tree depth')
plt.show()

#%%

# Function to plot the results
def plot_fitting_curves(results):
    results.plot("min_leaves")
    plt.ylabel("Accuracy")
    plt.xlabel("Min samples per leaf (inverse of complexity)")
    plt.show()
    
accuracy_on_train = []
accuracy_on_test = []

min_leaves = range(1,80,5)
for m in min_leaves:
    # the smaller min_samples_leaf the bigger is the tree (f.e leafs wiith only 6 children and grandchildren will not be allowed)
    model = DecisionTreeClassifier(min_samples_leaf=m, random_state=42)
    model.fit(X_train, y_train)
    # Evaluation in the same data used to train the model
    predictions_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, predictions_train,normalize=True, sample_weight=None)
    accuracy_on_train.append(accuracy_train)
    # Evaluation in previously unseen data
    predictions_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test,normalize=True, sample_weight=None)
    accuracy_on_test.append(accuracy_test)
    
results = pd.DataFrame({"min_leaves": min_leaves, "training": accuracy_on_train, "test":accuracy_on_test})
plot_fitting_curves(results)
# graph is basically the same as above, however, as the min samples param increases, the train accuracy will decrease (because tree will grow smaller)
# (Optimum is 42)

# Problem -> Overfitting to the testing data (also data leakage)

#%%

accuracy_on_train = []
accuracy_on_test = []

# specify criterion = 'entropy'
# specify max_depth (try out)
# min_samples_split (try out (probably something bigger than two, minimum))
# Reminder : min_samples_split specifies the minimum number of samples required to split an internal node, while min_samples_leaf specifies the minimum number of samples required to be at a leaf node.
# min_impurity_decrease (try out) (probably something bigger than .10)

min_impurity_dec = np.arange(0.0,0.9,0.1)
for m in min_impurity_dec:
    # the smaller min_samples_leaf the bigger is the tree (f.e leafs wiith only 6 children and grandchildren will not be allowed)
    model = DecisionTreeClassifier(min_impurity_decrease=m, random_state=42)
    model.fit(X_train, y_train)
    # Evaluation in the same data used to train the model
    predictions_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, predictions_train,normalize=True, sample_weight=None)
    accuracy_on_train.append(accuracy_train)
    # Evaluation in previously unseen data
    predictions_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test,normalize=True, sample_weight=None)
    accuracy_on_test.append(accuracy_test)
    
results = pd.DataFrame({"min_decrease": min_impurity_dec, "training": accuracy_on_train, "test":accuracy_on_test})
results.plot("min_decrease")
plt.ylabel("Accuracy")
plt.xlabel("Min impurity decrease (inverse of complexity)")
plt.show()
# graph is basically the same as above, however, as the min samples param increases, the train accuracy will decrease (because tree will grow smaller)
# (Optimum is 42)

# Problem -> Overfitting to the testing data (also data leakage)

#%%

# Narrow down range of optimal value
accuracy_on_train = []
accuracy_on_test = []

# specify criterion = 'entropy'
# specify max_depth (try out)
# min_samples_split (try out (probably something bigger than two, minimum))
# Reminder : min_samples_split specifies the minimum number of samples required to split an internal node, while min_samples_leaf specifies the minimum number of samples required to be at a leaf node.
# min_impurity_decrease (try out) (probably something bigger than .10)

min_impurity_dec = np.arange(0.0,0.2,0.05)
for m in min_impurity_dec:
    # the smaller min_samples_leaf the bigger is the tree (f.e leafs wiith only 6 children and grandchildren will not be allowed)
    model = DecisionTreeClassifier(min_impurity_decrease=m, random_state=42)
    model.fit(X_train, y_train)
    # Evaluation in the same data used to train the model
    predictions_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, predictions_train,normalize=True, sample_weight=None)
    accuracy_on_train.append(accuracy_train)
    # Evaluation in previously unseen data
    predictions_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, predictions_test,normalize=True, sample_weight=None)
    accuracy_on_test.append(accuracy_test)
    
results = pd.DataFrame({"min_decrease": min_impurity_dec, "training": accuracy_on_train, "test":accuracy_on_test})
results.plot("min_decrease")
plt.ylabel("Accuracy")
plt.xlabel("Min impurity decrease (inverse of complexity)")
plt.show()

#%%

#Combine the found hyperparameters and test via 10-fold cross validation
model = DecisionTreeClassifier(min_impurity_decrease=0.1, min_samples_leaf=40, random_state=42)
print(cross_val_score(model, X, y, cv=10))
#Achieves around 78% accuracy
cross_val_score(model, X, y, cv=10).mean()

#%%

model = DecisionTreeClassifier(min_samples_leaf=41, random_state=42)
print(cross_val_score(model, X, y, cv=10))
#Achieves around 81.3% accuracy
cross_val_score(model, X, y, cv=10).mean()

#%%

model = DecisionTreeClassifier(min_impurity_decrease=0.1, random_state=42)
print(cross_val_score(model, X, y, cv=10))
#Achieves around 78% accuracy
cross_val_score(model, X, y, cv=10).mean()

#%%

model = DecisionTreeClassifier(max_depth=3, random_state=42)
print(cross_val_score(model, X, y, cv=10))
#Achieves around 81.2% accuracy
cross_val_score(model, X, y, cv=10).mean()

#%%

# No difference between criterions
model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=41)
print(cross_val_score(model, X, y, cv=10))
#Achieves around 81% accuracy
cross_val_score(model, X, y, cv=10).mean()

#%%

# Use decision tree model with highest accuracy after 10-fold Cross-Validation -> min_samples_leaf=41
# and train it on the full dataset
optim_model = DecisionTreeClassifier(min_samples_leaf=41, random_state=42)
optim_model.fit(X,y)

#%%

from sklearn.model_selection import GridSearchCV

# Alternative approach, use best parameters using GridSearch

#finding best fit with gridsearch
param_grid = {'min_samples_leaf':np.arange(10,50,5),
              'min_samples_split':np.arange(10,50,5),
              'max_depth':np.arange(2,8),
              'min_weight_fraction_leaf':np.arange(0,0.5,0.1),
              'criterion':['gini','entropy']}
clf = tree.DecisionTreeClassifier()
search = GridSearchCV(clf, param_grid, scoring='average_precision')

search.fit(X,y)
print("params:",search.best_params_)
print("estimator :",search.best_estimator_ )
print("score :",search.best_score_ )

#%%

pred_opt_cv_train = optim_model.predict(X)

#%%

pred_opt_gs_model = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=20, min_samples_split=35, min_weight_fraction_leaf=0.0)
pred_opt_gs_train = pred_opt_gs_model.fit(X,y).predict(X)


#%%

print("Accuracy from whole train data with CV Optimization:",accuracy_score(y, pred_opt_cv_train, normalize=True, sample_weight=None))
print("Confusion Matrix", '\n', confusion_matrix(y, pred_opt_cv_train))
print("Classification Report:", '\n', classification_report(y, pred_opt_cv_train))

print("Accuracy from whole train data with GridSearch Optimization:",accuracy_score(y, pred_opt_gs_train, normalize=True, sample_weight=None))
print("Confusion Matrix", '\n', confusion_matrix(y, pred_opt_gs_train))
print("Classification Report:", '\n', classification_report(y, pred_opt_gs_train))

#%%

# Yields around 81.5 vs 83.5% accuracy -> Choose Grid Search parameters
# Predict value on Test Data

pred_opt_gs_test = pred_opt_gs_model.predict(test_df[features])

#%%

test_df['Survived'] = pred_opt_gs_test
Kaggle_submission = test_df
Kaggle_submission.drop(['Cabin','Pclass','Age','SibSp','Parch','Fare_zscore','male','Embarked_Q','Embarked_S','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_N','Cabin_T'], axis=1, inplace=True)
Kaggle_submission.head()
Kaggle_submission.to_csv("Kaggle_Submission.csv")

#%%

Kaggle_submission.head(20)

#%%

import csv

# Read the CSV file and store the modified data
with open('Kaggle_Submission.csv', 'r') as file:
    reader = csv.reader(file)
    data = [row[1:] for row in reader]  # Remove the first element in each row

# Write the modified data back to the CSV file
with open('Kaggle_Submission_cleaned.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)