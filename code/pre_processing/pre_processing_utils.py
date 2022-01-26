import h5py
import pandas as pd
import numpy as np 
import json
import scipy as sp
import nibabel as nib 
from glob import glob
import fnmatch
import os
os.chdir("../")
from individual_differences_utils import save_dict_greater_than_4gb, load_dict
os.chdir("pre_processing")

run_name_dict = {
    "REST1": "REST1_7T_PA",
    "REST2": "REST2_7T_AP",
    "REST3": "REST3_7T_PA",
    "REST4": "REST4_7T_AP",
    "MOVIE1": 'MOVIE1_CC1',
    "MOVIE2": 'MOVIE2_HO1',
    "MOVIE3": 'MOVIE3_CC2',
    "MOVIE4": 'MOVIE4_HO2'
}

def mk_TR_by_Feature(clip, feature_file): 

    if clip in run_name_dict.keys():
        run_name = run_name_dict[clip]
    hf = h5py.File(feature_file, 'r') 
    clip_features = hf.get(run_name)
    clip_features = np.array(clip_features)

    return clip_features

def get_visual_semantic_dropped_TRs(clip):
    run_name =  run_name_dict[clip]
    hf = h5py.File('../../data/7T_movie_resources/WordNetFeatures.hdf5', 'r')
    clip_features = hf.get(run_name)
    clip_features = np.array(clip_features)

    est_idx = hf.get(run_name + "_est")[()]
    val_idx = hf.get(run_name + "_val")[()]

    tr_df = pd.DataFrame((est_idx + val_idx))
    tr_idx = tr_df[tr_df.iloc[:,0] == 0].index.tolist()  
    return tr_idx

def offset_feature_matrix_by_TRs(feature_mat, offset_TRs = [1, 2, 3, 4, 5, 6, 7, 8]): # includes current TR
    TR_rows, feature_cols = feature_mat.shape
    vector_of_0s = pd.concat([pd.DataFrame(np.zeros(feature_cols).reshape(1, -1))]*offset_TRs[0], ignore_index=True)
    feature_mat_new = np.concatenate((vector_of_0s, feature_mat))
    feature_mat_offset = np.hstack((feature_mat, feature_mat_new[0:feature_mat.shape[0], :])) 
    
    for TR_num in offset_TRs[1:]:  
        vector_of_0s = pd.concat([pd.DataFrame(np.zeros(feature_cols).reshape(1, -1))]*TR_num, ignore_index=True) 
        feature_mat_new = np.concatenate((vector_of_0s, feature_mat))
        feature_mat_offset = np.hstack((feature_mat_offset, feature_mat_new[0:feature_mat_offset.shape[0], :]))
    
    return feature_mat_offset

def read_json_list(fileName):
    with open(fileName, "r") as fp:
        b = json.load(fp)
    return b

def get_TRs_with_visual_features(clip, 
                       start_stop_pads = (10, 5),
                       subj_list="../../data/train_subjects_list.npy"): # Contact us for access to train_subjects_list 
                        
    subj_list = np.load(subj_list, allow_pickle=True)
    
    if clip in run_name_dict.keys():
        run_name = run_name_dict[clip]

    if 'MOVIE' in clip:
        print("Removing TRs where no visual semantic features")
        hf = h5py.File('../../data/WordNetFeatures.hdf5', 'r') # File from HCP data from gallant lab's featurizations
        estIdx = hf.get(run_name + "_est")[()]
        valIdx = hf.get(run_name + "_val")[()]
        
        tr_df = pd.DataFrame((estIdx + valIdx).tolist())
        tr_idx = tr_df[tr_df.iloc[:,0] == 1].index.tolist()
    else: 
        print("This is not a movie 1-4 in HCP data. This function will not work.")

    return tr_idx

def make_TR_by_ROI_dict(clip, 
                        start_stop_pads = (10, 5),
                        subj_list="../../data/train_subjects_list.npy"): # Contact us for access to train_subjects_list 
                        
    subj_list = np.load(subj_list, allow_pickle=True)
    subjwise_ts_dict = {}

    # Get name clip is referred to in files 
    if clip in run_name_dict.keys():
        run_name = run_name_dict[clip]
    else: 
        print("This is not a movie or resting state 1-4 in HCP data. This function will not work.")

    if 'MOVIE' in clip:
        f_suffix = "_tfMRI_" + run_name + "_shen268_roi_ts.txt"
        data_dir = "../../data/participant_data/pre_processed_movie"
    elif "REST" in clip:
        f_suffix = "_rfMRI_" + run_name + "_shen268_roi_ts.txt"
        data_dir= "../../data/participant_data/pre_processed_rest"
               
    # Load data and if movie clip select desired TRs
    for s,subj in enumerate(subj_list):
        f_name = data_dir + subj + f_suffix

        run_data = pd.read_csv(f_name, sep='\t', header=None).dropna(axis=1)
        run_data = run_data.values # convert to np array
    
        # If movie clip subset TRs 
        if 'MOVIE' in clip:
            tr_idx = get_TRs_to_analyze(clip, start_stop_pads = (10, 5), subj_list="train_subjects_list.npy")
            run_data = run_data[tr_idx, :]

        # Zscore movie or rest clip (aka run)
        subjwise_ts_dict[subj] = sp.stats.zscore(run_data) 

    return subjwise_ts_dict

def get_voxels_masked_subset_zscored(nii_file_name, movie_idx, thin_mask):
    nii = nib.load(nii_file_name)
    nii_data = np.asanyarray(nii.dataobj).T
    nii_data_idx = nii_data[movie_idx ,:]
    nii_data_idx_masked = nii_data_idx[:, thin_mask]
    niiData_zscored = sp.stats.zscore(nii_data_idx_masked, axis=0)
    niiData_preprocessed = np.nan_to_num(niiData_zscored, nan=0.0)
    return niiData_preprocessed

def get_voxels_masked(nii_file_name, thin_mask):
    nii = nib.load(nii_file_name)
    nii_data = np.asanyarray(nii.dataobj).T
    nii_data_masked = nii_data[:, thin_mask]
    return nii_data_masked

# Check that all participants were presented the same subtasks in the same order in the motor stimulus 
def check_subtask_consistency(num_participants, data_file_rows, task_name): #can use this function for other HCP 3T tasks
    block_order_motor_task = pd.DataFrame(np.zeros((num_participants, data_file_rows)))
    count = 0 
    for d in glob("../../data/HCP_3T_Task/*"):
        if d != "../../data/HCP_3T_Task/pre_processed":
            for file in os.listdir(d + "../../data/MNINonLinear/Results/" + task_name + "/"):
                if fnmatch.fnmatch(file, "*_TAB.txt"):
                    data_file = pd.read_csv("../../data/HCP_3T_Task/" + d + "/MNINonLinear/Results/" + task_name + "/" + file, delimiter = "\t")
                    block_order_motor_task.iloc[count, :] = data_file["BlockType"]
                    count += 1
                
    ## check if any columns (subtasks) differ
    subset_block_order_motor_task = block_order_motor_task.iloc[0:count,:]
    if subset_block_order_motor_task[subset_block_order_motor_task.columns[subset_block_order_motor_task.apply(lambda s: len(s.unique()) > 1)]].shape[1] > 0:
        print("For at least 1 participant 1 of the subtasks differs from the other participants. Will need to further investigate the task to continue")
    else:
        print("Check complete, can proceed with pre-processing data")

def get_timing_RL(timing, first_task, data_file):
    start_time = "{}Cue.OnsetTime".format(first_task)
    start_values = data_file[start_time].dropna().unique()

    timing[sub_idx, 1] = start_values[0] 
    timing[sub_idx, 3] = start_values[1]

    trs_start_collecting = data_file["CountDownSlide.OnsetTime"] 
    timing[sub_idx, 4] =  trs_start_collecting[0]
    
    # fixation time - 1st fixation time and 3rd fixation time 
    fixation_onsets = data_file["Fixdot.OnsetTime"].dropna().unique()
    timing[sub_idx, 8] = fixation_onsets[0]
    timing[sub_idx, 9] = fixation_onsets[2]

    return timing

def get_timing_LR(timing, first_train_task, first_test_task, data_file):

    start_time_train = "{}Cue.OnsetTime".format(first_train_task)
    start_time_test = "{}Cue.OnsetTime".format(first_test_task)

    start_values_train = data_file[start_time_train].dropna().unique()
    start_values_test = data_file[start_time_test].dropna().unique()

    timing[sub_idx, 1] = start_values_train[0] 
    timing[sub_idx, 3] = start_values_test[1]

    trs_start_collecting = data_file["CountDownSlide.OnsetTime"] 
    timing[sub_idx, 4] =  trs_start_collecting[0]
    
    # fixation time - 2nd fixation time and 3rd fixation time 
    fixation_onsets = data_file["Fixdot.OnsetTime"].dropna().unique()
    timing[sub_idx, 8] = fixation_onsets[1]
    timing[sub_idx, 9] = fixation_onsets[2]

    return timing

def get_motor_task_cue_timing(subject_list, run, last_task): # function for RL or LR encoding of fMRI scan 
    timing = np.zeros((len(subject_list), 12))
    sub_without_motor = list()
    sub_no_tab_file = list()
    for sub_idx, sub in enumerate(subject_list):
        d = "../../data/HCP_3T_Task/{}".format(sub)
        timing[sub_idx, 0] = sub
        if os.path.isdir(d):
            
            for file in os.listdir("{dir_name}/MNINonLinear/Results/{run_name}/".format(dir_name = d, run_name = run)):
                if fnmatch.fnmatch(file, "*_TAB.txt"):
                    data_file = pd.read_csv("{dir_name}/MNINonLinear/Results/{run_name}/{file_name}".format(dir_name=d, 
                                                                        run_name = run, file_name = file), delimiter = "\t")
                    cue_idx = 0 
                    if run contains "RL":
                        timing = get_timing_RL(timing, "LeftHand", data_file)
                
                    elif run contains "LR":
                         timing = get_timing_LR(timing, "RightHand", "Tongue", data_file)

                else: # if tab file doesn't exist even though fMRI data exists 
                    sub_no_tab_file.append(sub)
                
            if os.path.isfile("{dir_name}/MNINonLinear/Results/{run_name}/EVs/{last_task_name}.txt".format(dir_name = d, 
                                                                run_name = run, last_task_name = last_task)): #ev file exists 
                last_task_file = pd.read_csv("{dir_name}/MNINonLinear/Results/{run_name}/EVs/{last_task_name}.txt".format(dir_name = d, 
                                                                 run_name = run, last_task_name = last_task), delimiter = "\t")
                timing[sub_idx, 2] = last_task_file.iloc[-1, 0]
            else: 
                print("sub {} no ev file".format(sub))
        else: 
            sub_without_motor.append(sub)
                    
    # rescale timing: need to remove the countDownSlide.OnsetTime which is when TRs started being recorded to make 
    # sure timing is in line with tr recording
    timing[:, 5] = (timing[:, 1] - timing[:, 4]) / 1000  #timing was in milliseconds --> seconds
    timing[:, 6] = (timing[:, 2] + 12)  # time was in seconds
    timing[:, 7] = (timing[:, 3] - timing[:, 4]) / 1000  #timing was in milliseconds 
    timing[:, 10] = (timing[:, 8] - timing[:, 4]) / 1000  #timing was in milliseconds 
    timing[:, 11] = (timing[:, 9] - timing[:, 4]) / 1000  #timing was in milliseconds 

    return timing, sub_without_motor, sub_no_tab_file 

def get_motor_task_cue_timing_from_evs(timing_file, subjects_no_tab_file, all_subjects, run, last_task_name):
    for sub in subjects_no_tab_file:
        sub_idx = all_subjects.index(sub) # get from idx in sub list
        d = "../../data/HCP_3T_Task/{}".format(sub)
        if os.path.isdir(d):
            cue_file = "{dir_name}/MNINonLinear/Results/{run_name}/EVs/cue.txt".format(dir_name = d, run_name = run)
            last_task_file = "{dir_name}/MNINonLinear/Results/{run_name}/EVs/{short_task_name}.txt".format(dir_name = d, run_name = run, short_task_name = last_task_name)
            if os.path.isfile(cue_file):
                cue = pd.read_csv(cue_file, delimiter = "\t", header = None)
                cue_times = cue.iloc[:, 0]
                timing_file[sub_idx, 1] = cue_times.iloc[0] 
                timing_file[sub_idx, 3] = cue_times.iloc[5]
                
            if os.path.isfile(last_task_file):
                last_task = pd.read_csv(last_task_file, delimiter = "\t")
                last_task_times = last_task.iloc[:, 0]
                timing_file[sub_idx, 2] = last_task_times.iloc[-1]
                
        # timing in ev files is already 0 indexed - as in 0 is start of when TRs collected 
        # to keep timing matrix consistent with matrix from get_motor_task_cue_timing function 
        # just copied these times to later cols 
        timing_file[sub_idx, 5] = timing_file[sub_idx, 1] 
        timing_file[sub_idx, 6] = timing_file[sub_idx, 2] + 12 
        timing_file[sub_idx, 7] = timing_file[sub_idx, 3] 

    return timing_file

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

      