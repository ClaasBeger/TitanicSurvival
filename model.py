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


