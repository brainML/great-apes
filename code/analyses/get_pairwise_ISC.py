import numpy as np 
import os 
os.chdir("../")
from individual_differences_utils import load_dict
os.chdir("analyses")

# Get ISC
def calculate_pairwise_ISC(train_subjects, num_ROIs, subjwise_ts_dict):
    ISC = np.zeros((len(train_subjects) * len(train_subjects), num_ROIs)) # (all subject pairs x rois)

    for sub_1_idx, sub_1 in enumerate(train_subjects): 
        tr_roi_sub_1 = subjwise_ts_dict[sub_1]

        for sub_2_idx, sub_2 in enumerate(train_subjects): 
            tr_roi_sub_2 = subjwise_ts_dict[sub_2]

            # Get pearson corr each roi sub_1 x each roi sub_2 
            pearson_corr_coef_matrix = np.corrcoef(tr_roi_sub_1, tr_roi_sub_2, rowvar= False) # sub_1[1],sub_2[1] (index [1,num_roi + 1]) and sub_2[1],sub_1[1] (index [num_roi + 1,1])

            # Select corr between same ROI for the two subjects (i.e. sub_1 roi_1 and sub_2 roi_1) 
            for sub_1_roi_1 in np.arange(num_ROIs):
                ISC[(sub_1_idx * len(train_subjects)) + sub_2_idx, sub_1_roi_1] = pearson_corr_coef_matrix[sub_1_roi_1, sub_1_roi_1 + num_ROIs]
    return ISC


# Remove index/rows of data where corr same sub with itself, as including this would artificially increase ISC value
def remove_corr_sub_with_itself(train_subjects, sub_pair_by_roi_mat):
    idx_corr_same_subject = []
    for sub_1_idx in np.arange(len(train_subjects)): 
        idx_corr_same_subject.append((sub_1_idx * len(train_subjects)) + sub_1_idx)
    
    sub_pair_by_roi_mat_updated = np.delete(sub_pair_by_roi_mat, 
                                          idx_corr_same_subject, axis = 0)
    
    return sub_pair_by_roi_mat_updated




train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
subjwise_ts_dict_all_subs = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscoredPerRunAndThenAcross4Runs_movie") # key = subject, value = TR x ROI matrix
# ^ created in pre_processing/D_get_TRs_by_ROIs_to_analyze_movie.py

subjwise_ts_dict_train_subs = {your_key: subjwise_ts_dict_all_subs[your_key] for your_key in train_subs}

ISC_mat = calculate_pairwise_ISC(train_subs, 268, subjwise_ts_dict_train_subs)
ISC_mat_without_same_sub_corr = remove_corr_sub_with_itself(train_subs, ISC_mat)

np.save("ISC_7T_movie_data.npy", ISC_mat_without_same_sub_corr)