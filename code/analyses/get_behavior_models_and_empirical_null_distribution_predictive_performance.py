from analyses_utils import read_json_list, get_updated_train_subs_drop_subjects_without_enocding_model_performance, get_permuted_subject_order
from behavior_models import get_behavior_model_predictive_performance_for_unpermuted_subjects_and_empirical_null_distribution
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", dest="model_type", type=str, help="Specify type of encoding model (ie. APE, language_stimulus, visual_stimulus)")
parser.add_argument("--brain_region_type", dest="brain_region_type", type=str, help="Specify type of brain region (ie. voxel, ROI)")
parser.add_argument("--HCP_task", dest="HCP_task", type=str, help="Specify HCP task name (ie. movie, motor)")
parser.add_argument("--minimum_percent_significantly_predicted_subjects", dest="minimum_percent_significantly_predicted_subjects", type=float, default = .33, help="""Specify 
                    the percent of subjects that must have predictive performance significantly higher than chance, for a brain region to be included in behavior model in 
                    the feature matrix.""")
parser.add_argument("--num_of_permutations", dest="num_of_permutations", type=int, default = 10001, help="Specify number of permutations")


args = parser.parse_args()
model_type = args.model_type
brain_region_type = args.brain_region_type
HCP_task = args.HCP_task
minimum_percent_significantly_predicted_subjects = args.minimum_percent_significantly_predicted_subjects
num_of_permutations = args.num_of_permutations

def get_behavior_data_for_selected_subjects(selected_subjects):
  behav_data = pd.read_csv('../data/all_behav.csv',
                                 dtype={'Subject': 'str'}) # Downloaded from HCP
  behav_data.set_index("Subject", inplace=True)
  behav_data_selected_subjects = behav_data_train.loc[selected_subjects]

  return behav_data_selected_subjects

# Get label for each participants mother. This will be used to ensure that entire families are left out, to prevent family 
# similarity from artifically inflating performance of behavior model
def get_family_label_for_selected_subjects(selected_subjects):
  res_behav_data = pd.read_csv("../../data/res_behav_data.csv", dtype={'Subject':str}) # Requires HCP restricted data access
  res_behav_data = res_behav_data.set_index("Subject")
  mothers = res_behav_data.loc[selected_subjects, "Mother_ID"]
  mothers = mothers.to_frame()

  return mothers

def get_encoding_model_predictive_performance_matrix_for_behavior_model(brain_region_type, model_type, HCP_task, train_subs, minimum_percent_significantly_predicted_subjects): 
# ^ This matrix will be used as the features to predict a participants behavior measure scores from  
  num_subjects_sig_predicted_per_brain_region = np.load("../../data/encoding_models/num_subjects_sig_predicted_per_brain_region_{REGION}_level_{MODEL}_{TASK}".format( REGION = brain_region_type, 
                                                MODEL = model_type, TASK = HCP_task))
  sub_brain_region_corr = np.zeros((len(train_subs), len(num_subjects_sig_predicted_per_brain_region))) # Number of subjects by number of brain regions
  for sub_idx, sub in enumerate(train_subs): 
    sub_encoding_model_predictive_performance = np.load("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=sub, REGION = brain_region_type, 
              MODEL = model_type, TASK = HCP_task))
    sub_brain_region_predictive_performance[sub_idx, :] = sub_encoding_model_predictive_performance
  minimum_num_significantly_predicted_subjects = np.ceil(minimum_percent_significantly_predicted_subjects * len(train_subs))
  
  return sub_brain_region_predictive_performance[:, np.argwhere(num_subjects_sig_predicted_per_brain_region >= minimum_num_significantly_predicted_subjects).squeeze()]

# Drop subjects without behavior measure of interest from behavior data, predictive performance and family labels. 
def get_subset_data_for_subjects_with_behavior_measure(behavior_measure, behavior_data, predictive_performance_matrix, family_labels, train_subs, sub_missing_data):
  # Check if subjects missing behavior measure, appears as nans
  if np.sum(np.isnan(np.asarray(behavior_data[behavior_measure]))) == 0: # No nan's
      return behavior_data[behavior_measure], predictive_performance_matrix, family_labels, sub_missing_data
  else: 
        behavior_data_without_dropped_subjects = behavior_data[behavior_measure].drop(behavior_data.loc[pd.isna(behavior_data[behavior_measure]), :].index)
        sub_to_drop = []
        for sub in behavior_data.loc[pd.isna(behavior_data[behavior_measure]), :].index:
            sub_to_drop.append(train_subs.index(sub)) # Row position of missing subject in data to subset
            sub_missing_data.append(sub) # Subject number of missing subject

        predictive_performance_matrix_without_dropped_subjects = np.delete(predictive_performance_matrix, sub_to_drop, axis = 0)
        sub_to_keep = np.delete(behavior_data[behavior_measure].index, sub_to_drop)

        family_labels_without_dropped_subjects = family_lables.loc[family_labels.index.isin(sub_to_keep)]

  return behavior_data_without_dropped_subjects, predictive_performance_matrix_without_dropped_subjects, family_labels_without_dropped_subjects, sub_missing_data

train_subs = read_json_list(train_subjects_list) # Contact us for access to train_subjects_list (the list of participants we have selected for the development set)
train_subs, sub_missing_data = get_updated_train_subs_drop_subjects_without_encoding_model_performance(train_subs, brain_region_type, model_type, HCP_task)
behavior_data = get_behavior_data_for_selected_subjects(train_subs)
family_labels = get_family_label_for_selected_subjects(train_subs)
predictive_performance_matrix = get_encoding_model_predictive_performance_matrix_for_behavior_model(brain_region_type, model_type, HCP_task, train_subs, minimum_percent_significantly_predicted_subjects)
permuted_subject_matrix = get_permuted_subject_order(train_subs, num_permutations) # For null distribution
cognitive_measures = ["CogFluidComp_AgeAdj", "CogTotalComp_AgeAdj", "CogCrystalComp_AgeAdj", "VSPLOT_TC", "PicSeq_AgeAdj", "CardSort_AgeAdj",
                      "Flanker_AgeAdj", "ListSort_AgeAdj", "ProcSpeed_AgeAdj",  "SCPT_SEN",  "IWRD_TOT", "PicVocab_AgeAdj", "ReadEng_AgeAdj",
                      "PMAT24_A_CR", "DDisc_AUC_40K"]

for behavior_measure in cognitive_measures:
  subset_behavior_data, subset_predictive_performance_matrix, subset_family_labels, sub_missing_data = get_subset_data_for_subjects_with_behavior_measures(behavior_measure, behavior_data, 
    predictive_performance_matrix, family_labels, sub_missing_data)
  behavior_data_2d = subset_behavior_data.reshape(len(subset_behavior_data), 1) # Reshape behavior_data

  predictive_performance = get_behavior_model_predictive_performance_for_unpermuted_subjects_and_empirical_null_distribution(behavior_data_2d, subset_predictive_performance_matrix, 
    subset_family_labels, permuted_subject_matrix, sub_missing_data, num_of_permutations)
  np.save("../../data/behavior_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=sub, REGION = brain_region_type, MODEL = model_type, TASK = HCP_task), predictive_performance)
