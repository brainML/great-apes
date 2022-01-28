from encoding_models import encoding_model_return_predictions
from analyses_utils import read_json_list
import numpy as np
import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import load_dict

train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)

subjwise_ts_dict_all_subs = load_dict("../../data/subjwise_dict_TR_by_ROI_matrix_zscoredPerRunAndThenAcross4Runs_rest") # key = subject, value = TR x ROI matrix
# ^ created in pre_processing/D_get_TRs_by_ROIs_to_analyze_rest.py

subjwise_ts_dict_train_subs = {your_key: subjwise_ts_dict_all_subs[your_key] for your_key in train_subs}

for s,sub in enumerate(train_subs):
    subject_TRs_by_ROI = subjwise_ts_dict_train_subs[subj]
    feature_mat = np.array([subjwise_ts_dict_train_subs[k] for k in subjwise_ts_dict_train_subs if k != sub]).mean(axis = 0)
    correlations, predictions, lambdas = encoding_model_return_predictions(subject_TRs_by_ROI, feature_mat, n_folds = 4, splits = [900, 900 + 900, 900 + 900 + 900, 900 + 900 + 900 + 900]) 

    np.save("../../data/encoding_models/sub_{SUB}_corr_ROI_level_APE_rest.npy".format(SUB=sub), correlations)
    np.save("../../data/encoding_models/sub_{SUB}_predictions_ROI_level_APE_rest.npy".format(SUB=sub), predictions)
    np.save("../../data/encoding_models/sub_{SUB}_lambdas_ROI_level_APE_rest.npy".format(SUB=sub), lambdas)
