#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:13:21 2018

@author: bking
"""

import pandas as pd
#import numpy as np
#from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("dataset/train_origin.csv")
test_df = pd.read_csv("dataset/test_origin.csv")

# Remove column with one value or not make sense
columns_to_remove = [col for col in train_df.columns if train_df[col].nunique() == 1]
columns_to_remove += ['date','sessionId','visitStartTime']

# Filter
train_df_ = train_df.drop(columns=columns_to_remove)
test_df_ = test_df.copy()
test_df_ = test_df_.loc[:,train_df_.columns]
test_df_ = test_df_.drop(['totals.transactionRevenue'],axis=1)
# Fill N/A
train_df_ = train_df_.fillna(0)
test_df_ = test_df_.fillna(0)

# Type Conversion
fullVisitorId_list = train_df_['fullVisitorId'].unique()
fullVisitorId_dict = {k:v for v,k in enumerate(fullVisitorId_list)}

train_df_['fullVisitorId'] = train_df_['fullVisitorId'].apply(lambda x: fullVisitorId_dict.get(x))
test_df_['fullVisitorId'] = test_df_['fullVisitorId'].apply(lambda x: fullVisitorId_dict.get(x))

# Label encoder
object_col = [i for i in train_df_.columns if train_df_[i].dtype == 'O']
for col in object_col:
    print (col)
    train_unique = train_df_[col].unique()
    train_dict = {k:v for v,k in enumerate(train_unique)}
    train_df_[col] = train_df_[col].apply(lambda x: train_dict.get(x))
    test_df_[col] = test_df_[col].apply(lambda x: train_dict.get(x,-1))

# Export data
train_df_.to_csv("data/train_pre.csv")
test_df_.to_csv("data/test_pre.csv")
   
