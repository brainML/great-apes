import pandas as pd
import numpy as np 
import scipy as sp
import argparse
import cpmJenn as cpm
from sklearn.preprocessing import StandardScaler
import h5py
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import individual_differences_utils as utils
import pickle as pkl
import scipy.stats as stats
from sklearn.model_selection import KFold
import argparse

def predict_leave_one_family_out(scores, correlations, family_list): 
    preds = np.zeros_like(scores)
    nsubs = correlations.shape[0] 
    for idx, i in enumerate(np.unique(family_list["Mother_ID"])):
        subdata = np.hstack([correlations, np.ones((nsubs,1))]) # adding intercept 
        train_data = subdata[family_list["Mother_ID"] != i] 
        train_scores = scores[family_list["Mother_ID"] != i] 
        weights,__ = utils.cross_val_ridge(train_data,train_scores)
        preds[family_list["Mother_ID"] == i] = subdata[family_list["Mother_ID"] == i].dot(weights)
        
    return preds


def predict_true_and_permuted_leave_one_family_out(scores, correlations, family_list, motorAPE, drop_subs, n_perm = 10000):
    n = correlations.shape[0]
    permuted_scores = np.zeros((n,n_perm))

    permuted_scores[:,0] = scores 

    for j in range(1,n_perm):
        if j%500 ==0:
            print(j)
        permute_order = permuted_sub_labels[:, j] 
        if drop_subs != None:
            for value_to_drop in drop_subs:
                permute_order = np.delete(permute_order, np.argwhere(permute_order == int(value_to_drop)))

        if motorAPE:
            for sub in ['126931', '745555']:
                permute_order = np.delete(permute_order, np.argwhere(permute_order == int(sub)))
        permute_order = [str(i) for i in permute_order]
        permuted_scores[:,j]= scores.reindex(permute_order).squeeze()
    preds = predict_leave_one_family_out(permuted_scores,correlations, family_list)
    corr_permuted = utils.corr(preds, permuted_scores)
    corr_unpermuted = corr_permuted[0]
    return corr_permuted, np.percentile(corr_permuted,5),np.percentile(corr_permuted,95)


