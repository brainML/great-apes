import numpy as np 
import scipy as sp
from pre_processing_utils import read_json_list, check_subtask_consistency, get_timing_RL, get_timing_LR, get_motor_task_cue_timing, get_motor_task_cue_timing_from_evs, get_voxels_masked

def subset_motor_task_into_task_repeats(run_data, motor_timing, sub_idx, zscore = True):
    
    start_repeat1 = np.ceil(motor_timing[sub_idx, 5] / .72 ) # .72 is TR length https://www.mail-archive.com/hcp-users@humanconnectome.org/msg00616.html
    start_repeat2 = np.ceil((motor_timing[sub_idx, 7]) / .72)
    stop_repeat1 = start_repeat2 - 15  # subtract 15 seconds based off canonical HRF #15 seconds worth of TRs skipped is 21 TRs
    
    stop_repeat2 = run_data.shape[0]
    repeat1_subjects_tr_by_voxel = run_data[int(start_repeat1):int(stop_repeat1),:] 
    repeat2_subjects_tr_by_voxel = run_data[int(start_repeat2):int(stop_repeat2),:] 
    
    if zscore: 
        repeat1_subjects_tr_by_voxel = sp.stats.zscore(repeat1_subjects_tr_by_voxel)
        repeat2_subjects_tr_by_voxel = sp.stats.zscore(repeat2_subjects_tr_by_voxel)
    
    return repeat1_subjects_tr_by_voxel, repeat2_subjects_tr_by_voxel


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
# Subset voxel data: select thin mask voxels, split into train and test based off TRs
thin_mask = np.load("../../data/cortical_thin_mask_7_21_21_2mmMNI_HCP.npy")

# Load participants 
train_subjects = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)

# Split each participant's motor run into train and test: use first 1/2 to train and second 1/2 to test 
# Subset voxel data
for sub_idx, sub in enumerate(train_subjects):
    if sub in list(set(sub_no_lr_motor_data + sub_no_rl_motor_data)): #skip participants that are missing at least 1 run of motor data
        continue 
    else: 
        RL_file = "../../data/HCP_3T_Task/{SUB}/MNINonLinear/Results/tfMRI_MOTOR_RL/tfMRI_MOTOR_RL.nii.gz".format(SUB=sub)
        LR_file = "../../data/HCP_3T_Task/{SUB}/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz".format(SUB=sub)

        RL_data = get_voxels_masked(RL_file)    
        LR_data = get_voxels_masked(LR_file)

        # now subset - as two repeats per run: zscore each repeat as treat each repeat as a run 
        repeat1_rl, repeat2_rl = subset_motor_task_into_task_repeats(RL_data, motor_rl_timing_complete, sub_idx)
        repeat1_lr, repeat2_lr = subset_motor_task_into_task_repeats(LR_data, motor_lr_timing_complete, sub_idx)

        # combine - and zscore across the 4 repeats as treat as 4 runs 
        motor_1_2 = np.vstack((repeat1_rl, repeat2_rl))
        motor_1_2_3 = np.vstack((motor_1_2, repeat1_lr))
        motor_1_2_3_4 = np.vstack((motor_1_2_3, repeat2_lr))
        motor_zscored = sp.stats.zscore(motor_1_2_3_4, axis = 0)

        np.save("./../data/HCP_3T_Motor_Voxel_Space/pre_processed/sub_{SUB}_zscored_per_run_and_across_4_runs_thin_mask".format(SUB = sub), motor_zscored)

