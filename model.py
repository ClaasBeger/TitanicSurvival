import pandas as pd

import torch

import numpy as np

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

#%%

# Check for categorical variables

train_df.applymap(np.isreal).all(0)

#%%

# Drop obsolete categories

train_df = train_df.drop(labels=['Name'], axis=1)


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







