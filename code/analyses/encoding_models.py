import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import cross_val_ridge_predictions, corr, get_CV_ind_specificSplits,load_dict 
import numpy as np 
import scipy as sp

def encoding_model_return_predictions(data, features, n_folds = 4, splits = [131, 131+126, 131+126+111, 131+126+111+146]):
    # ^ split defaults are for HCP 3T motor task
    n,v = data.shape
    p = features.shape[1]
    corrs = np.zeros((n_folds,v))
    ind = get_CV_ind_specificSplits(splits, n_folds)
    preds_all = np.zeros_like(data)
    lambdas_all = np.zeros((4, v))
    for i in range(n_folds):
        train_data = np.nan_to_num(sp.stats.zscore(data[ind!=i]))
        train_features = np.nan_to_num(sp.stats.zscore(features[ind!=i]))
        test_data = np.nan_to_num(sp.stats.zscore(data[ind==i]))
        test_features = np.nan_to_num(sp.stats.zscore(features[ind==i]))
        preds, lambdas = cross_val_ridge_predictions(train_features, train_data, test_features, method = 'sklearn_kernel_ridge')
        preds_all[ind==i] = preds
        lambdas_all[i, :] = lambdas
    corrs = corr(preds_all, data)
    return corrs, preds_all, lambdas_all  

def encoding_model_return_empirical_null_distribution_predictive_performance(predictions, true_data, num_brain_regions, shuffled_time,
    num_permutations = 10000, n_folds = 4, splits = [769, 769 + 795, 769 + 795 + 763,769 + 795 + 763 + 778]):
    # ^ split defaults are for HCP 7T movie task

    # For each fold get predictions and true_data 
    fold_predictions = {}
    fold_true_data = {}
    for i in np.arange(n_folds):
        if i == 0:
            fold_predictions[i] = predictions[0:splits[i],:]
            fold_true_data[i] = true_data[0:splits[i],:]
        elif i > 0:
            fold_predictions[i] = predictions[splits[i-1]:splits[i],:]
            fold_true_data[i] = true_data[splits[i-1]:splits[i],:]

    # For each permutation get null distribution predictive performace
    null_dist_predictive_performance = {}
    shuffled_time_dict = load_dict(shuffled_time) 
    # ^ dict where key = fold number, value = matrix where rows are number of time points in fold, cols are number of permutations
    for permutation_num in np.arange(num_permutations):

        corrs = np.zeros((n_folds, num_brain_regions))
        for i in np.arange(n_folds):
            # Get permutated time (row) order 
            fold_shuffled_time_index = shuffled_time_dict[i]
            fold_permuted_row_index = fold_shuffled_time_index[:,permutation_num].astype(int)

            # Permute predictions using permuted time order
            fold_permuted_predictions = fold_predictions[i][fold_permuted_row_index, :]

            # Correlate permuted predictions and true data
            corrs[i, :] = corr(fold_permuted_predictions, fold_true_data[i])

        null_dist_predictive_performance["permutation_{}".format(permutation_num)] = corrs.mean(0)

    return null_dist_predictive_performance
