import os 
import pandas as pd
import numpy as np 
import scipy as sp
from pre_processing_utils import read_json_list, check_subtask_consistency, get_timing_RL, get_timing_LR, get_motor_task_cue_timing, get_motor_task_cue_timing_from_evs

def subset_train_test_data_motor_tr_by_roi_data(motor_timing, subject_list, run, zscore = True):
    train_subjects_tr_by_roi = {}
    test_subjects_tr_by_roi = {}
    for sub_idx, sub in enumerate(subject_list):
    
        f = "../../data/participant_data/pre_processed_motor/{subject}_tfMRI_{run_name}_shen268_roi_ts.txt".format(subject = sub,
                                                                                            run_name = run)
        if os.path.isfile(f):
            run_data = pd.read_csv(f, sep='\t', header=None).dropna(axis=1)
            start_train = np.ceil(motor_timing[sub_idx, 5] / .72 ) # .72 is TR length https://www.mail-archive.com/hcp-users@humanconnectome.org/msg00616.html
            start_test = np.ceil((motor_timing[sub_idx, 7]) / .72)
            stop_train = start_test - 15  # subtract 15 seconds based off canonical HRF
            
            stop_test = run_data.shape[0]
            train_subjects_tr_by_roi[sub] = run_data.iloc[int(start_train):int(stop_train),:].reset_index(drop = True)
            test_subjects_tr_by_roi[sub] = run_data.iloc[int(start_test):int(stop_test),:].reset_index(drop = True)
            
            if zscore: 
                train_subjects_tr_by_roi[sub] = sp.stats.zscore(train_subjects_tr_by_roi[sub])
                test_subjects_tr_by_roi[sub] = sp.stats.zscore(test_subjects_tr_by_roi[sub])
                
    return train_subjects_tr_by_roi, test_subjects_tr_by_roi
    

# Check motor stimulus subtasks consistent across participants
check_subtask_consistency(182, 117, "tfMRI_MOTOR_RL")
check_subtask_consistency(182, 117, "tfMRI_MOTOR_LR")

# Get timing of motor stimulus subtasks, and which participants don't have motor fMRI data or a descriptive tab file
train_subjects = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
motor_rl_timing, sub_no_rl_motor_data, sub_no_rl_tab_file = get_motor_task_cue_timing(train_subjects, "tfMRI_MOTOR_RL", "lf") 
motor_lr_timing, sub_no_lr_motor_data, sub_no_lr_tab_file = get_motor_task_cue_timing(train_subjects, "tfMRI_MOTOR_LR", "rf") 

# For participants without a descriptive tab file, need to get the timing from their HCP EV file
motor_rl_timing_complete = get_motor_task_cue_timing_from_evs(motor_rl_timing, sub_no_rl_tab_file, train_subjects, "tfMRI_MOTOR_RL", "lf") 
motor_lr_timing_complete = get_motor_task_cue_timing_from_evs(motor_lr_timing, sub_no_lr_tab_file, train_subjects, "tfMRI_MOTOR_LR", "rf") 

# Split each participant's motor run into train and test: use first 1/2 to train and second 1/2 to test 
# Subset ROI data
train_rl, test_rl = subset_train_test_data_motor_tr_by_roi_data(motor_rl_timing_complete, 
                                                train_subjects, "MOTOR_RL", zscore = True)
test_lr, train_lr = subset_train_test_data_motor_tr_by_roi_data(motor_lr_timing_complete, 
                                                train_subjects, "MOTOR_LR", zscore = True)

# Save 
save_dict(train_rl, "../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_LR_train")
save_dict(train_lr, "../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_LR_train")
save_dict(test_rl, "../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_RL_test")
save_dict(test_lr, "../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_LR_test")