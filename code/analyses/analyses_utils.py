import numpy as np 
import os
import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import save_dict_greater_than_4gb, load_dict, save_dict

def read_json_list(fileName):
    with open(fileName, "r") as fp:
        b = json.load(fp)
    return b

def get_subjects_missing_3t_fmri_data(subject_list, run): 
    sub_without_run_data = list()
    for sub_idx, sub in enumerate(subject_list):
        d = "../../data/HCP_3T_Task/{}".format(sub)
        if os.path.isdir("{dir_name}/MNINonLinear/Results/{run_name}".format(dir_name = d, run_name = run)):
        else: 
            sub_without_run_data.append(sub)
                    
    return sub_without_run_data

# Subset TR x voxel matrices to select some voxels for batches of subjects
def subset_voxels_for_batches_of_subjects(train_subjects, start_sub, stop_sub, start_vox, stop_vox, task, scanner_resolution, skip_subs = []): 
    for idx, sub in enumerate(train_subjects):
        if idx >= start_sub and idx < stop_sub:
            if sub not in skip_subs:
                subjwise_ts_dict_4runs[sub] = np.load("../../data/HCP_{TESLA}_{TASK}_Voxel_Space/pre_processed/sub_{SUB}_zscored_per_run_and_across_4_runs_thin_mask".format(TESLA = scanner_resolution, 
                    TASK = task, SUB = sub))[:, start_vox:stop_vox]

    # Save voxel subset across all subjects 
    save_dict_greater_than_4gb(subjwise_ts_dict_4runs, "../../data/HCP_{TESLA}_{TASK}_Voxel_Space/pre_processed/{TASK}_voxel_{START}_{STOP}_train_sub_{SUBSTART}_{SUBSTOP}".format(TESLA = scanner_resolution, 
        TASK = task, START = start_vox, STOP = stop_vox, SUBSTART = start_sub, SUBSTOP = stop_sub))
    
# For each batch of voxels, save the average of all possible groups of leave one participant out
def average_subset_voxels_with_leave_one_subject_out(num_voxels, train_subjects, step_size, subject_step_size, task, scanner_resolution, skip_subs = []): 
    for start_vox in np.arange(0, num_voxels, step_size):
        all_subs = {}
        if start_vox + step_size < num_voxels:
            stop_vox = start_vox + step_size
        elif start_vox + step_size > num_voxels:
            stop_vox = num_voxels
        for start_sub in np.arange(0, len(train_subjects), subject_step_size):
            all_subs.update(load_dict("../../data/HCP_{TESLA}_{TASK}_Voxel_Space/pre_processed/{TASK}_voxel_{START}_{STOP}_train_sub_{SUBSTART}_{SUBSTOP}".format(TESLA = scanner_resolution, TASK = task, 
                START = start_vox, STOP = stop_vox, SUBSTART = start_sub, SUBSTOP = start_sub + subject_step_size)))
        for train_subject in train_subjects:
            if sub not in skip_subs:
                avg_tr_voxels_leave_one_sub_out = np.nanmean(np.array([all_subs[k] for k in all_subs if k != train_subject]), axis = 0)
                save_dict_greater_than_4gb(avg_tr_voxels_leave_one_sub_out, "../../data/HCP_{TESLA}_{TASK}_Voxel_Space/pre_processed/{TASK}_voxel_{START}_{STOP}_averaged_with_sub_{SUB}_left_out".format(TESLA = scanner_resolution,
                    TASK = task, START = start_vox, STOP=stop_vox, SUB = train_subject))
            
# For all voxels, save the average of all possible groups of leave one participant out
def get_matrix_with_average_for_each_voxel_with_leave_one_subject_out(num_TRS, num_voxels, train_subjects, step_size, task, scanner_resolution, skip_subs = []):
    for s,sub in enumerate(train_subjects):
        if sub not in skip_subs:
            avg_tr_voxels_leave_one_sub_out = np.zeros((num_TRs, num_voxels))

            for start_vox in np.arange(0, num_voxels, step_size):

                if start_vox + step_size < num_voxels:
                    stop_vox = start_vox + step_size
                elif start_vox + step_size > num_voxels:
                    stop_vox = num_voxels

                avg_tr_voxels_leave_one_sub_out[:, start_vox:stop_vox] = load_dict("../../data/HCP_{TESLA}_{TASK}_Voxel_Space/pre_processed/{TASK}_voxel_{START}_{STOP}_averaged_with_sub_{SUB}_left_out".format(TESLA = scanner_resolution,
                    TASK = task, START = start_vox, STOP= stop_vox, SUB = sub))

            save_dict_greater_than_4gb(avg_tr_voxels_leave_one_sub_out, "../../data/HCP_{TESLA}_{TASK}_Voxel_Space/pre_processed/tr_by_voxel_averaged_with_sub_{SUB}_left_out_{TASK}".format(TESLA = scanner_resolution,
                    TASK = task, SUB=sub))

def get_average_encoding_model_predictive_performance(train_subs, num_brain_regions, brain_region_type, model_type, task): # average is across all participants per brain region
    sub_by_brain_region = np.zeros((len(train_subs), num_brain_regions))
    for s,sub in enumerate(train_subs):
        sub_encoding_model_predictive_performance = np.load("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=sub, 
                                                    REGION = brain_region_type, MODEL = model_type, TASK = task))
        sub_by_brain_region[s, :] = sub_encoding_model_predictive_performance
    average_encoding_model_predictive_performance = np.nanmean(sub_by_brain_region, axis = 0)

    return average_encoding_model_predictive_performance

def get_encoding_model_predictive_performance_variability(train_subs, num_brain_regions, brain_region_type, model_type, task): 
# ^ aka coefficient of variation, variability is across participants per brain region
    sub_by_brain_region = np.zeros((len(train_subs), num_brain_regions))
    for s,sub in enumerate(train_subs):
        sub_encoding_model_predictive_performance = np.load("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=sub, 
                                                    REGION = brain_region_type, MODEL = model_type, TASK = task))
        sub_by_brain_region[s, :] = sub_encoding_model_predictive_performance

    performance_variability = np.nanstd(sub_by_brain_region, axis = 0) / np.abs(np.nanmean(sub_by_brain_region, axis = 0))

    return performance_variability

# For each permutation get permuted order for all of the time blocks in the data.  
def get_permutated_order_of_time_blocks(num_of_blocks, num_permutations):
    shuffle_order = np.zeros((np.int(np.ceil(num_of_blocks)), num_permutations))
    for permutation_num in np.arange(num_permutations):
        shuffle_order[:, permutation_num] = np.random.choice(np.arange(np.ceil(num_10_tr_blocks)), np.int(np.ceil(num_10_tr_blocks)), replace = False)
    
    return shuffle_order

# Get time block start and stop row index. This assumes that each time point is a row. 
# This is standard for fMRI data where each row = 1 TR (repitition time). 
def get_start_and_end_row_index_of_time_blocks(num_of_blocks, length_of_time_block):
    fractional, whole = math.modf(num_of_blocks) 

    start_row_index_of_block = {}
    end_row_index_of_block = {}

    for block in np.arange(np.int(np.ceil(num_of_blocks))):
        start_row_index_of_block[block] = block * length_of_time_block
        if block == np.floor(num_of_blocks):
            end_row_index_of_block[block] = (block * length_of_time_block) + np.round(fractional * length_of_time_block)  - 1 # as row indexes are 0 indexed, need -1
        else: 
            end_row_index_of_block[block] = (block * length_of_time_block) + (length_of_time_block -1)
    
    return start_row_index_block, end_row_index_of_block

def get_permuted_order_all_time_points(num_time_points, num_permutations, shuffle_order, start_tr_of_block, end_tr_of_block):
    permuted_order_all_time_points = np.zeros((num_time_points, num_permutations))
    for permutation_idx in np.arange(permuted_order.shape[1]): 
        time_point_count = 0 
        for idx_in_shuffle_order, block_num in enumerate(shuffle_order[:, permutation_idx]):
            start_time_point = start_time_point_of_block[block_num]
            end_time_point = end_time_point_of_block[block_num]

            time_points_in_block = np.arange(start_time_point, end_time_point + 1)
            shuffle_order_all_time_points[time_point_count : time_point_count + len(time_points_in_block), permutation_idx] = time_points_in_block
            time_point_count += len(time_points_in_block) # doing this because 1 of the blocks is only 5 TRs 
    
    return permuted_order_all_time_points

def get_permuted_time_dictionary(task, num_of_folds = 4, num_permutations = 10000, num_time_points_per_fold = None, num_time_points_per_block = None):
    if os.path.isfile("../../data/shuffled_time_dictionary_{TASK}".format(TASK = task)): # File already exists
        permuted_time_dict = {}
        for fold in np.arange(num_of_folds):
            permuted_time_dict[fold] = load_dict("../../data/shuffled_time_dictionary_{TASK}_fold_{FOLD}".format(TASK = task, FOLD = fold))

    else: 
        if task == "movie": 
            num_time_points_per_fold = [769, 795, 763, 778] 
            num_time_points_per_block = 20
        elif task == "motor": 
            num_time_points_per_fold = [131, 126, 146, 111] 
            num_time_points_per_block = 28
        elif task == "rest": 
            num_time_points_per_fold = [900, 900, 900, 900] 
            num_time_points_per_block = 20
        else:
            if num_time_points_per_fold == None or num_time_points_per_block == None:
                print("""Task that was specified does not have defaults for num_time_points_per_fold and num_time_points_per_block in function.
                Need to specify these parameters. """)

        permuted_time_dict = {}
        for fold in np.arange(num_of_folds):
            num_of_blocks_in_fold = num_time_points_per_fold[fold] / num_time_points_per_block
            permutated_order_time_blocks = get_permutated_order_of_time_blocks(num_of_blocks_in_fold, num_permutations)
            start_of_blocks, end_of_blocks = get_start_and_end_row_index_of_time_blocks(num_of_blocks_in_fold, num_time_points_per_block)
            permuted_time_dict[fold] = get_permuted_order_all_time_points(num_time_points_per_fold[fold], num_permutations, permutated_order_time_blocks, 
                                  start_of_blocks, end_of_blocks)
            save_dict(permuted_time_dict[fold], "../../data/shuffled_time_dictionary_{TASK}_fold_{FOLD}".format(TASK = task, FOLD = fold))

    return permuted_time_dict

