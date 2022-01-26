import numpy as np
from pre_processing_utils import read_json_list, subset_voxels_for_batches_of_subjects, average_subset_voxels_with_leave_one_subject_out, get_matrix_with_average_for_each_voxel_with_leave_one_subject_out

# Memory constraints of HPC required subsetting of voxels and participants to eventually obtain feature matrices for APE model
# Step_size and subject_step_size can be increased if the user's system has more memory. Code can also be simiplified to skip subsetting if memory allows.
step_size = 10000 
subject_step_size = 10 
total_num_voxels = 131906 # in HCP 7T movie data
total_num_TRs = 3105 # in HCP movie task
train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
magnetic_field_strength = "7T" # in HCP 7T movie data
task_name = "Movie"

for start_voxel in np.arange(0, total_num_voxels, step_size): 
	if start_voxel + step_size < total_num_voxels:
        stop_voxel = start_voxel + step_size
    elif start_voxel + step_size > total_num_voxels:
        stop_voxel = total_num_voxels
	for start_subject in np.arange(0, len(train_subs), subject_step_size): 
		stop_subject = start_subject + subject_step_size
		subset_voxels_for_batches_of_subjects(train_subs, start_subject, stop_subject, start_voxel, stop_voxel, task_name, magnetic_field_strength)

average_subset_voxels_with_leave_one_subject_out(total_num_voxels, train_subs, step_size, subject_step_size, task_name, magnetic_field_strength)
get_matrix_with_average_for_each_voxel_with_leave_one_subject_out(total_num_TRs, total_num_voxels, train_subs, step_size, task_name, magnetic_field_strength)