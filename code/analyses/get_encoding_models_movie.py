from encoding_models import enocoding_model_return_predictions
from analyses_utils import read_json_list
import numpy as np
os.chdir("../")
from individual_differences_utils import load_dict
os.chdir("analyses")
import argparse

arser = argparse.ArgumentParser()
parser.add_argument("--model_type", dest="model_type", type=str, help="Specify type of encoding model (ie. APE, language_stimulus, visual_stimulus)")
parser.add_argument("--brain_region_type", dest="brain_region_type", type=str, help="Specify type of brain region (ie. voxel, ROI)")

args = parser.parse_args()

model_type = args.model_type
brain_region_type = args.brain_region_type

def get_brain_activity_to_predict(brain_region, sub):
    if brain_region == "ROI":
        subjwise_ts_dict_all_subs = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscoredPerRunAndThenAcross4Runs_movie")
        # ^ created in pre_processing/D_get_TRs_by_ROIs_to_analyze_movie.py
        subject_TRs_by_brain_region = subjwise_ts_dict_all_subs[sub]

    elif brain_region == "voxel":
        subject_TRs_by_brain_region = np.load("../../data/HCP_7T_Movie_Voxel_Space/pre_processed/sub_{SUB}_zscored_per_run_and_across_4_runs_thin_mask".format(SUB = sub))
        # ^ created in pre_processing/CC_get_per_subject_TRs_by_voxels_movie.py

    return subject_TRs_by_brain_region

# Get feature space matrix predicting brain activity from
def get_feature_space_matrix(train_subjects, model, brain_region, sub):
    if model == "APE":
        if brain_region == "ROI":
            subjwise_ts_dict_all_subs = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscoredPerRunAndThenAcross4Runs_movie") # key = subject, value = TR x ROI matrix
            # ^ created in pre_processing/D_get_TRs_by_ROIs_to_analyze_movie.py
            subjwise_ts_dict_train_subs = {your_key: subjwise_ts_dict_all_subs[your_key] for your_key in train_subjects}
            feature_mat = np.array([subjwise_ts_dict_train_subs[k] for k in subjwise_ts_dict_train_subs if k != sub]).mean(axis = 0)
        elif brain_region == "voxel":
            feature_mat = load_dict("../../data/HCP_7T_Movie_Voxel_Space/pre_processed/tr_by_voxel_averaged_with_sub_{SUB}_left_out_Movie".format(SUB=sub))
            # ^ created in get_APE_feature_space_matrices_tr_by_voxels_movie.py

    elif model == "language_stimulus":
        feature_mat = np.load("../../data/movie_language_semantic_features.npy") 

    elif model == "visual_stimulus":
        feature_mat = np.load("../../data/movie_visual_semantic_features.npy") # created in pre_processing/get_visual_feature_matrix.py

    return feature_mat


train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
for s,sub in enumerate(train_subs):
    brain_activity_to_predict = get_brain_activity_to_predict(brain_region_type, sub)
    feature_space_matrix = get_feature_space_matrix(train_subs, model_type, brain_region_type, sub) 
    correlations, predictions, lambdas = encoding_model_return_predictions(brain_activity_to_predict, feature_space_matrix, n_folds = 4, splits = [769, 769 + 795, 769 + 795 + 763,769 + 795 + 763 + 778]) 

    np.save("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_movie.npy".format(SUB=sub, REGION = brain_region_type, MODEL = model_type), correlations)
    np.save("../../data/encoding_models/sub_{SUB}_predictions_{REGION}_level_{MODEL}_movie.npy".format(SUB=sub, REGION = brain_region_type, MODEL = model_type), predictions)
    np.save("../../data/encoding_models/sub_{SUB}_lambdas_{REGION}_level_{MODEL}_movie.npy".format(SUB=sub, REGION = brain_region_type, MODEL = model_type), lambdas)