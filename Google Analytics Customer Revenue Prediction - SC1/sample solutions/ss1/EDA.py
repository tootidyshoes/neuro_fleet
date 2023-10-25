#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:20:02 2018

@author: bking
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load data
train_df = pd.read_csv("data/train_pre.csv",index_col=0)


def corr_matrix(df_truth):
    # Compute the correlation matrix
    corr = df_truth.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    

corr_matrix(train_df)