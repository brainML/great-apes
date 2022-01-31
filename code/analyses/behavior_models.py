import numpy as np 
import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import corr, cross_val_ridge_predictions

def get_behavior_model_predictive_performance_leave_one_family_out(behavior_scores, features, family_list): 
    all_predictions = np.zeros_like(behavior_scores)
    num_subs = features.shape[0] 
    for idx, i in enumerate(np.unique(family_list["Mother_ID"])):
        sub_features = np.hstack([features, np.ones((num_subs,1))]) # adding intercept 
        train_features = sub_features[family_list["Mother_ID"] != i] 
        test_features = sub_features[family_list["Mother_ID"] == i] 
        train_behavior_scores = behavior_scores[family_list["Mother_ID"] != i] 
        predictions, lambdas = cross_val_ridge_predictions(np.nan_to_num(train_features), np.nan_to_num(train_behavior_scores), np.nan_to_num(test_features), method = 'sklearn_kernel_ridge')
        all_predictions[family_list["Mother_ID"] == i] = predictions
        
    return all_predictions

def get_behavior_model_predictive_performance_for_unpermuted_subjects_and_empirical_null_distribution(behavior_scores, features, family_list, permuted_sub_labels, 
    subjects_missing_data = None, num_permutations = 10001):
    num_subs = features.shape[0]
    behavior_scores_to_predict = np.zeros((num_subs, num_permutations))
    behavior_scores_to_predict[:,0] = behavior_scores 

    for j in range(1,num_permutations):
        if j%500 ==0:
            print(j)
        permute_order = permuted_sub_labels[:, j-1] 
        if subjects_missing_data != None:
            for sub_to_drop in subjects_missing_data:
                permute_order = np.delete(permute_order, np.argwhere(permute_order == int(sub_to_drop)))
        permute_order = [str(i) for i in permute_order]
        behavior_scores_to_predict[:,j]= behavior_scores.reindex(permute_order).squeeze()
    predictions = get_behavior_model_predictive_performance_leave_one_family_out(behavior_scores_to_predict, features, family_list)
    predictive_performances = corr(permuted_predictions, behavior_scores_to_predict) 
    
    return predictive_performances


