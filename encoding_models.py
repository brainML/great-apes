
import individual_differences_utils as utils
import numpy as np 
import scipy as sp

def encoding_model_returnPredictions(data, features, n_folds = 4, splits = [131, 131+126, 131+126+111, 131+126+111+146]):
    n,v = data.shape
    p = features.shape[1]
    corrs = np.zeros((n_folds,v))
    ind = utils.get_CV_ind_specificSplits(splits, n_folds)
    preds_all = np.zeros_like(data)
    weights_all = {}
    lambdas_all = np.zeros((4, v))
    for i in range(n_folds):
        train_data = np.nan_to_num(zscore(data[ind!=i]))
        train_features = np.nan_to_num(zscore(features[ind!=i]))
        test_data = np.nan_to_num(zscore(data[ind==i]))
        test_features = np.nan_to_num(zscore(features[ind==i]))
        preds, lambdas = utils.cross_val_ridge_predictions(train_features, train_data, test_features, method = 'sklearn_kernel_ridge')
        preds_all[ind==i] = preds
        lambdas_all[i, :] = lambdas
        #print('fold {}'.format(i))
    corrs = utils.corr(preds_all, data)
    return corrs, preds_all,  lambdas_all 