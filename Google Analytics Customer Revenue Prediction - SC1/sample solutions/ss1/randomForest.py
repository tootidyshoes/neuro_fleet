#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:24:01 2018

@author: bking
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
#from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("data/train_pre.csv",index_col=0)
test_df = pd.read_csv("data/test_pre.csv",index_col=0)

# Prepare data
train_y = train_df['totals.transactionRevenue']
train_x = train_df.drop(['totals.transactionRevenue'],axis=1)

# Random Forest Regression with HyperParameter Tuning
rf = RandomForestRegressor()
#param = {'max_depth':[3,6,10,15,20,None]}
param = {'max_depth':[3,10]}
rf_cv = GridSearchCV(rf,param,cv=2,verbose=True,scoring='neg_mean_squared_log_error')

rf_cv.fit(train_x, train_y)

# save the model to disk
filename = 'model/rf_cv.sav'
pickle.dump(rf_cv, open(filename, 'wb'))

best_model = rf_cv.best_estimator_
