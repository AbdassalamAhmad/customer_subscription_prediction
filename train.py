#!/usr/bin/env python
# coding: utf-8

# Importing
import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
import xgboost as xgb



# reading the two datasets (training & testing)
df = pd.read_csv('bank-full.csv',sep=';')



#data preparation
df.y = (df.y == 'yes').astype(int) #changing our output (y) from categorical to numerical


#feature engineering for both datasets
df['balance'] = df['balance'].mask(df['balance'] < 0, 0)
balance_logs = np.log1p(df.balance)
df['balance_logs'] = balance_logs


del df['balance']# we delete balance and replacing it with balance_logs because it will do better in the model.
df=df[['age', 'job', 'marital', 'education', 'default', 'balance_logs', 'housing', 'loan',
       'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
       'poutcome', 'y']]# reordering DataFrame


cols = ['age', 'job', 'marital', 'education', 'default', 'balance_logs', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome']

y_train=df.y.values



#Training
from sklearn.feature_extraction import DictVectorizer
train_dict = df[cols].fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict)

#training XGB_best_model
features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)


#watchlist = [(dtrain, 'train'), (dval, 'val')]
#%%capture output

xgb_params = {
    'eta': 0.1, 
    'max_depth': 5,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,)

# Save the model

output_file = 'model_1.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')




