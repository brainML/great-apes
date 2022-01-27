from encoding_models import enocoding_model_return_predictions
from analyses_utils import read_json_list
import numpy as np
os.chdir("../")
from individual_differences_utils import load_dict
os.chdir("analyses")
import argparse

arser = argparse.ArgumentParser()
parser.add_argument("--brain_region_type", dest="brain_region_type", type=str, help="Specify type of brain region (ie. voxel, ROI)")

args = parser.parse_args()

brain_region_type = args.brain_region_type

def get_brain_activity_to_predict(brain_region, subjwise_dict_1, subjwise_dict_2, subjwise_dict_3, subjwise_dict_4, sub):
    if brain_region == "ROI":
        # Combine the 4 repeats, as treat as 4 runs 
        subjwise_ts_dict_all_subs = combine_4_runs_of_data_for_each_sub(subjwise_dict_1, subjwise_dict_2, subjwise_dict_3, subjwise_dict_4)
        subject_TRs_by_brain_region = subjwise_ts_dict_all_subs[sub]

    elif brain_region == "voxel":
        subject_TRs_by_brain_region = np.load("./../data/HCP_3T_Motor_Voxel_Space/pre_processed/sub_{SUB}_zscored_per_run_and_across_4_runs_thin_mask".format(SUB = sub))
        # ^ created in pre_processing/CC_get_per_subject_TRs_by_voxels_motor.py
    return subject_TRs_by_brain_region

def combine_4_runs_of_data_for_each_sub(dict_1, dict_2, dict_3, dict_4):
    dict_all_runs_all_subs = {}
    for sub in subjwise_ts_dict_all_subs_1.keys():   
        temp_mat_0 = np.vstack((dict_1[sub], dict_2[sub]))
        temp_mat_1 = np.vstack((temp_mat_0, dict_3[sub]))
        temp_mat_2 = np.vstack((temp_mat_1, dict_4[sub]))
        subjwise_ts_dict_all_subs[sub] = temp_mat_2
    return dict_all_runs_all_subs 

# Get feature space matrix predicting brain activity from
def get_feature_space_matrix(train_subjects, brain_region, subjwise_dict_1, subjwise_dict_2, subjwise_dict_3, subjwise_dict_4, sub): 
    if brain_region == "ROI":
        # Combine the 4 repeats, as treat as 4 runs 
        subjwise_ts_dict_all_subs = combine_4_runs_of_data_for_each_sub(subjwise_dict_1, subjwise_dict_2, subjwise_dict_3, subjwise_dict_4)
        subjwise_ts_dict_train_subs = {your_key: subjwise_ts_dict_all_subs[your_key] for your_key in train_subjects}
        feature_mat = np.array([subjwise_ts_dict_train_subs[k] for k in subjwise_ts_dict_train_subs if k != sub]).mean(axis = 0)

    elif brain_region == "voxel":
        feature_mat = load_dict("../../data/HCP_3T_Motor_Voxel_Space/pre_processed/tr_by_voxel_averaged_with_sub_{SUB}_left_out_Motor".format(SUB=sub))
        # ^ created in get_APE_feature_space_matrices_tr_by_voxels_motor.py
    return feature_mat


# Load dicts where key = sub, value = TR x ROI matrix. Need if ROI level analysis and to get number of TRs per run for both analysis types.
subjwise_ts_dict_all_subs_1 = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_RL_train") 
subjwise_ts_dict_all_subs_2 = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_RL_test")
subjwise_ts_dict_all_subs_3 = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_LR_train")
subjwise_ts_dict_all_subs_4 = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscored_per_participant_MOTOR_LR_test")
# ^ created in pre_processing/D_get_TRs_by_ROIs_to_analyze_motor.py

# Get TR length of runs to know how to split TRs when concatenated
example_subject = list(subjwise_ts_dict_all_subs_1.keys())[0]
run_1_TRs = subjwise_ts_dict_all_subs_1[example_subject].shape[0]
run_2_TRs = subjwise_ts_dict_all_subs_2[example_subject].shape[0]
run_3_TRs = subjwise_ts_dict_all_subs_3[example_subject].shape[0]
run_4_TRs = subjwise_ts_dict_all_subs_4[example_subject].shape[0]

train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
for s,sub in enumerate(train_subs):
    brain_activity_to_predict = get_brain_activity_to_predict(brain_region_type, subjwise_ts_dict_all_subs_1, subjwise_ts_dict_all_subs_2, subjwise_ts_dict_all_subs_3, 
        subjwise_ts_dict_all_subs_4, sub)
    feature_space_matrix = get_feature_space_matrix(train_subs, brain_region_type, subjwise_ts_dict_all_subs_1, subjwise_ts_dict_all_subs_2, subjwise_ts_dict_all_subs_3, 
        subjwise_ts_dict_all_subs_4, sub) 
    correlations, predictions, lambdas = encoding_model_return_predictions(brain_activity_to_predict, feature_space_matrix, n_folds = 4, splits = [run_1_TRs, 
        run_1_TRs+run_2_TRs, run_1_TRs+run_2_TRs+run_3_TRs, run_1_TRs+run_2_TRs+run_3_TRs+run_4_TRs]) 

    np.save("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_APE_motor.npy".format(SUB=sub, REGION = brain_region_type), correlations)
    np.save("../../data/encoding_models/sub_{SUB}_predictions_{REGION}_level_APE_motor.npy".format(SUB=sub, REGION = brain_region_type), predictions)
    np.save("../../data/encoding_models/sub_{SUB}_lambdas_{REGION}_level_APE_motor.npy".format(SUB=sub, REGION = brain_region_type), lambdas)

