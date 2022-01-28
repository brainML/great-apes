import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import load_dict, save_dict
from encoding_models import encoding_model_return_empirical_null_distribution_predictive_performance
from analyses_utils import read_json_list, get_permuted_time_dictionary, get_brain_activity_to_predict, get_subjects_missing_3t_fmri_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", dest="model_type", type=str, help="Specify type of encoding model (ie. APE, language_stimulus, visual_stimulus)")
parser.add_argument("--brain_region_type", dest="brain_region_type", type=str, help="Specify type of brain region (ie. voxel, ROI)")
parser.add_argument("--HCP_task", dest="HCP_task", type=str, help="Specify HCP task name (ie. movie, motor, rest)")
parser.add_argument("--num_of_permutations", dest="num_of_permutations", type=int, default = 10000, help="Specify number of permutations")
parser.add_argument("--start_sub_idx", dest="start_sub_idx", type=int, default = 0, help="""Specify start subject index. Permutations can take a 
					while, therefore if compute resouces allow multiple jobs to be run in parallel it is suggested to run 1 subject per job.""")
parser.add_argument("--stop_sub_idx", dest="stop_sub_idx", type=int, default = 1, help="""Specify stop subject index. Permutations can take a 
					while, therefore if compute resouces allow multiple jobs to be run in parallel it is suggested to run 1 subject per job.""")

args = parser.parse_args()

model_type = args.model_type
brain_region_type = args.brain_region_type
HCP_task = args.HCP_task
num_of_permutations = args.num_of_permutations
start_sub_idx = args.start_sub_idx
stop_sub_idx = args.stop_sub_idx

permuted_time_dictionary = get_permuted_time_dictionary(HCP_task)
train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
num_of_brain_regions = np.load("../../data/encoding_models/sub_{SUB}_predictions_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=train_subs[0], 
							REGION = brain_region_type, MODEL = model_type, TASK = HCP_task)).shape[1] 

# Get midding subjects and fold splits specific to HCP task analyzing
subjects_missing_data = []
if HCP_task == "movie":
	task_splits = [769, 769 + 795, 769 + 795 + 763, 769 + 795 + 763 + 778]
elif HCP_task == "motor":
	task_splits = [131, 131+126, 131+126+111, 131+126+111+146]
	# Get which participants don't have motor fMRI data or a descriptive tab file
	sub_no_rl_motor_data = get_subjects_missing_3t_fmri_data(train_subs, "tfMRI_MOTOR_RL") 
	sub_no_lr_motor_data = get_subjects_missing_3t_fmri_data(train_subs, "tfMRI_MOTOR_LR") 
	subjects_missing_data = list(set(sub_no_lr_motor_data + sub_no_rl_motor_data))# in HCP 3T motor data
elif HCP_task == "rest":
	task_splits = [900, 900 + 900, 900 + 900 + 900, 900 + 900 + 900 + 900]

for s,sub in enumerate(train_subs): 
	if s >= start_sub_idx and s < stop_sub_idx:
		if sub not in subjects_missing_data:
		    true_brain_activity = get_brain_activity_to_predict(brain_region_type, sub)
			predicted_brain_activity = np.load("../../data/encoding_models/sub_{SUB}_predictions_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=sub, 
									REGION = brain_region_type, MODEL = model_type, TASK = HCP_task))
			null_distribution_predictive_performance = encoding_model_return_empirical_null_distribution_predictive_performance(predicted_brain_activity, true_brain_activity, 
				num_of_brain_regions, permuted_time_dictionary, num_permutations = num_of_permutations, n_folds = 4, 
				splits = task_splits)
			save_dict(null_distribution_predictive_performance, "../../data/encoding_model_null_distributions/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}".format(SUB=sub, 
									REGION = brain_region_type, MODEL = model_type, TASK = HCP_task))