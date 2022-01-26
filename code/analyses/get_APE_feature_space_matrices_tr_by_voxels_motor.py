import numpy as np
from analyses_utils import read_json_list, subset_voxels_for_batches_of_subjects, average_subset_voxels_with_leave_one_subject_out, get_matrix_with_average_for_each_voxel_with_leave_one_subject_out

# Memory constraints of HPC required subsetting of voxels and participants to eventually obtain feature matrices for APE model
# Step_size and subject_step_size can be increased if the user's system has more memory. Code can also be simiplified to skip subsetting if memory allows.
step_size = 10000 
subject_step_size = 10 
total_num_voxels = 67427 # in HCP 3T motor data 
total_num_TRs = 514 # in HCP motor task 
magnetic_field_strength = "3T" # in HCP 3T motor data
task_name = "Motor"

train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)

# Get timing of motor stimulus subtasks, and which participants don't have motor fMRI data or a descriptive tab file
sub_no_rl_motor_data = get_subjects_missing_3t_fmri_data(train_subs, "tfMRI_MOTOR_RL") 
sub_no_lr_motor_data = get_subjects_missing_3t_fmri_data(train_subs, "tfMRI_MOTOR_LR") 
subjects_missing_data = list(set(sub_no_lr_motor_data + sub_no_rl_motor_data))# in HCP 3T motor data

for start_voxel in np.arange(0, total_num_voxels, step_size): 
	if start_voxel + step_size < total_num_voxels:
        stop_voxel = start_voxel + step_size
    elif start_voxel + step_size > total_num_voxels:
        stop_voxel = total_num_voxels
	for start_subject in np.arange(0, len(train_subs), subject_step_size): 
		stop_subject = start_subject + subject_step_size
		subset_voxels_for_batches_of_subjects(train_subs, start_subject, stop_subject, start_voxel, stop_voxel, task_name, magnetic_field_strength, subjects_missing_data)

average_subset_voxels_with_leave_one_subject_out(total_num_voxels, train_subs, step_size, subject_step_size, task_name, magnetic_field_strength, subjects_missing_data)
get_matrix_with_average_for_each_voxel_with_leave_one_subject_out(total_num_TRs, total_num_voxels, train_subs, step_size, task_name, magnetic_field_strength, subjects_missing_data)