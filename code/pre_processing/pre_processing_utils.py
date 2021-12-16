import h5py
import pandas as pd
import numpy as np 

offsetFeatureMatrixByTRs

run_name_dict = {
    "MOVIE1": 'MOVIE1_CC1',
    "MOVIE2": 'MOVIE2_HO1',
    "MOVIE3": 'MOVIE3_CC2',
    "MOVIE4": 'MOVIE4_HO2'
}

def mk_TR_by_Feature(clip, feature_file): 

    if clip in run_name_dict.keys():
        feature_name = run_name_dict[clip]
    hf = h5py.File(feature_file, 'r') 
    clip_features = hf.get(feature_name)
    clip_features = np.array(clip_features)

    return clip_features

def get_visual_semantic_dropped_TRs(clip):
    feature_name =  run_name_feature_dict[clip]
    hf = h5py.File('../data/7T_movie_resources/WordNetFeatures.hdf5', 'r')
    clip_features = hf.get(feature_name)
    clip_features = np.array(clip_features)

    est_idx = hf.get(feature_name + "_est")[()]
    val_idx = hf.get(feature_name + "_val")[()]

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

