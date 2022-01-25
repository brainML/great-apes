import numpy as np
import scipy as sp
import os
from pre_processing_utils import get_TRs_with_visual_features, read_json_list, get_voxels_masked_zscored

# Get TRs to analyze
input_kwargs ={
    "start_stop_pads": (10, 5),
    "subj_list": "../../data/train_subjects_list.npy", # Contact us for access to train_subjects_list 
}

M1_tr_idx = get_TRs_with_visual_features(clip="MOVIE1",**input_kwargs)
M2_tr_idx = get_TRs_with_visual_features(clip="MOVIE2",**input_kwargs) 
M3_tr_idx = get_TRs_with_visual_features(clip="MOVIE3",**input_kwargs)
M4_tr_idx = get_TRs_with_visual_features(clip="MOVIE4",**input_kwargs)

# Get voxels of interst (thin mask) and subjects of interest
thin_mask = np.load("../../data/thin_mask_1.6mm_MNI.npy")
train_subjects = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)

# Apply thin mask and zscore data 
movie_name_1 = "tfMRI_MOVIE1_7T_AP"
movie_name_2 = "tfMRI_MOVIE2_7T_PA"
movie_name_3 = "tfMRI_MOVIE3_7T_PA"
movie_name_4 = "tfMRI_MOVIE4_7T_AP"

for sub_idx, sub in enumerate(train_subjects):
    for file1 in os.listdir("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}".format(SUB = sub, MOVIE = movie_name_1)):
        zscored_1 = get_voxels_masked_zscored("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}/{FILE}".format(SUB = sub, MOVIE = movie_name_1, FILE = file1), M1_tr_idx)
        
    for file2 in os.listdir("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}".format(SUB = sub, MOVIE = movie_name_2)):
        zscored_2 = get_voxels_masked_zscored("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}/{FILE}".format(SUB = sub, MOVIE = movie_name_2, FILE = file2), M2_tr_idx)
        
    for file3 in os.listdir("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}".format(SUB = sub, MOVIE = movie_name_3)):
        zscored_3 = get_voxels_masked_zscored("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}/{FILE}".format(SUB = sub, MOVIE = movie_name_3, FILE = file3), M3_tr_idx)
        
    for file4 in os.listdir("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}".format(SUB = sub, MOVIE = movie_name_4)):
        zscored_4 = get_voxels_masked_zscored("../../data/HCP_7T_Movie_Voxel_Space/{SUB}/MNINonLinear/Results/{MOVIE}/{FILE}".format(SUB = sub, MOVIE = movie_name_4, FILE = file4), M4_tr_idx)
        
    # Concatenate movie clips together
    movie_1_2 = np.vstack((zscored_1, zscored_2))
    movie_1_2_3 = np.vstack((movie_1_2, zscored_3))
    movie_1_2_3_4 = np.vstack((movie_1_2_3, zscored_4))
    movies_zscored = sp.stats.zscore(movie_1_2_3_4, axis = 0)
    
    np.save("../../data/HCP_7T_Movie_Voxel_Space/pre_processed/sub_{SUB}_zscored_per_run_and_across_4_runs_thin_mask".format(SUB = sub), movies_zscored)
