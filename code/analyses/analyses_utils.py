import numpy as np 
import os
import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import save_dict_greater_than_4gb, load_dict

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