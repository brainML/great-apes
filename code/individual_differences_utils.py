# Functions to estimate cost for each lambda, by voxel:
from __future__ import division                                              

from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time 
import scipy as sp 
from sklearn.kernel_ridge import KernelRidge
import pickle as pkl
import pandas as pd

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

def save_dict(dictionary, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(dictionary, f)

def save_dict_greater_than_4gb(dictionary, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(dictionary, f, protocol=4)

def get_sub_by_roi(data_to_reshape, train_subs):
    pred_performance = np.zeros((len(train_subs),268))
    for subIdx in np.arange(0, len(train_subs)):
        pred_performance[subIdx, :] = data_to_reshape[subIdx* 268 : (subIdx *268) + 268]
    return pred_performance

def drop_subjects_without_behavior(behavior_data, predictive_performance, behavior_data_all):
    if np.sum(np.isnan(np.asarray(behavior_data))) == 0: 
        return behavior_data, predictive_performance, train_subs
    else: 
        behav_without_nans = behavior_data.drop(behavior_data_all.loc[pd.isna(behavior_data), :].index)
        sub_to_drop = []
        for sub in behavior_data_all.loc[pd.isna(behavior_data), :].index:
            sub_to_drop.append(train_subs.index(sub))
        dropped_sub_predictive_performance = np.delete(predictive_performance,sub_to_drop, axis = 0)
        sub_to_keep = np.delete(behavior_data.index, sub_to_drop)
        return behav_without_nans, dropped_sub_predictive_performance, sub_to_keep
    
def drop_subjects_without_behavior_3T(behavior_data, predictive_performance, behavior_data_all, train_subs_3T):
    if np.sum(np.isnan(np.asarray(behavior_data))) == 0: #no nan's
        return behavior_data, predictive_performance, train_subs_3T
    else: 
        
        behav_without_nans = behavior_data.drop(behavior_data_all.loc[pd.isna(behavior_data), :].index)
        sub_to_drop = []
        for sub in behavior_data_all.loc[pd.isna(behavior_data), :].index:
            sub_to_drop.append(train_subs_3T.index(sub))
        dropped_sub_predictive_performance = np.delete(predictive_performance,sub_to_drop, axis = 0)
        sub_to_keep = np.delete(behavior_data.index, sub_to_drop)
        return behav_without_nans, dropped_sub_predictive_performance, sub_to_keep

def corr(X,Y):
    return np.mean(sp.stats.zscore(X)*sp.stats.zscore(Y),0)

def kernel_ridge_sklearn(X, Y, lmbda): # jenn added 7/2/21
    clf = KernelRidge(alpha=lmbda)
    clf.fit(X,Y)
    return clf.dual_coef_

def ridge(X,Y,lmbda):
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def kernel_ridge_by_lambda_sklearn(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        clf = KernelRidge(alpha=lmbda)
        clf.fit(X,Y)
        error[idx] = 1 -  R2( clf.predict(Xval), Yval) 
    return error

def kernel_ridge_sklearn_predictions(X, Y, lmbda, Xval): 
    clf = KernelRidge(alpha=lmbda)
    clf.fit(X,Y)
    return clf.predict(Xval) 

def cross_val_ridge(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
    
    ridge_1 = dict(plain = ridge_by_lambda,
                   sklearn_kernel_ridge = kernel_ridge_by_lambda_sklearn)[method]
    ridge_2 = dict(plain = ridge,
                   sklearn_kernel_ridge = kernel_ridge_sklearn)[method]
    
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
       
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
        
    if do_plot:
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0)
    weights = np.zeros((train_features.shape[1],train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): 
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])


def cross_val_ridge_predictions(train_features, train_data, test_features, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
    
    ridge_1 = dict(plain = ridge_by_lambda,
                   sklearn_kernel_ridge = kernel_ridge_by_lambda_sklearn)[method]
    ridge_2 = dict(plain = ridge,
                   sklearn_kernel_ridge = kernel_ridge_sklearn_predictions)[method]
    
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
        if icv%3 ==0:
            print(icv)
    if do_plot:
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0)
    predictions = np.zeros((test_features.shape[0],train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): 
        idx_vox = argmin_lambda == idx_lambda
        if np.sum(idx_vox) > 0:
            predictions[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox], lambdas[idx_lambda], test_features)
    if do_plot:
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return predictions, np.array([lambdas[i] for i in argmin_lambda])


def get_CV_ind_specificSplits(vectorSplits, n_folds):
    n = vectorSplits[n_folds - 1]
    ind = np.zeros((n))
    for i in range(0,n_folds):
        #print("i ", i)
        if i == 0: 
            start = 0 
        else: 
            start = vectorSplits[i -1]
        if i == (n_folds -1):
            end = n 
        else: 
            end = vectorSplits[i]
        ind[start:end] = i 
    return ind
