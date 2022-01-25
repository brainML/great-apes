import h5py
import pandas as pd
import numpy as np 
import json
import scipy as sp

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
    hf = h5py.File('../data/7T_movie_resources/WordNetFeatures.hdf5', 'r')
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

def get_TRs_to_analyze(clip, 
                       start_stop_pads = (10, 5),
                       subj_list="train_subjects_list.npy"): # Contact us for access to train_subjects_list 
                        
    subj_list = np.load(subj_list, allow_pickle=True)
    
    if clip in run_name_dict.keys():
        run_name = run_name_dict[clip]

    if 'MOVIE' in clip:
        print("Removing TRs where no visual semantic features")
        hf = h5py.File('../data/WordNetFeatures.hdf5', 'r') # File from HCP data from gallant lab's featurizations
        estIdx = hf.get(run_name + "_est")[()]
        valIdx = hf.get(run_name + "_val")[()]
        
        tr_df = pd.DataFrame((estIdx + valIdx).tolist())
        tr_idx = tr_df[tr_df.iloc[:,0] == 1].index.tolist()
    else: 
        print("This is not a movie 1-4 in HCP data. This function will not work.")

    return tr_idx

def make_TR_by_ROI_dict(clip, 
                        start_stop_pads = (10, 5),
                        subj_list="train_subjects_list.npy"): # Contact us for access to train_subjects_list 
                        
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
