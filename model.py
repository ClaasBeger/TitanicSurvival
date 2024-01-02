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

train_df.drop(labels='Name', axis=1)

#%%

# One Hot Encode Gender

one_hot = pd.get_dummies(train_df['Sex'], drop_first=True)
train_df = train_df.join(one_hot)
