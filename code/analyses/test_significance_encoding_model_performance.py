import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import load_dict, save_dict
from analyses_utils import get_empirical_pvalue_one_sided_permutation_test, pvalue_multiple_hypothesis_benjamini_hochberg_fdr_correction, 
read_json_list, get_subjects_missing_3t_fmri_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", dest="model_type", type=str, help="Specify type of encoding model (ie. APE, language_stimulus, visual_stimulus)")
parser.add_argument("--brain_region_type", dest="brain_region_type", type=str, help="Specify type of brain region (ie. voxel, ROI)")
parser.add_argument("--HCP_task", dest="HCP_task", type=str, help="Specify HCP task name (ie. movie, motor, rest)")
parser.add_argument("--alpha", dest="alpha", type=float, default = 0.05, help="Specify alpha for one sided permutation test")

args = parser.parse_args()

model_type = args.model_type
brain_region_type = args.brain_region_type
HCP_task = args.HCP_task
alpha = args.alpha

train_subs = read_json_list(train_subjects_list) # contact us for access to train_subjects_list (the list of participants we have selected for the development set)
num_brain_regions = np.load("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB= train_subs[0], 
                                    REGION = brain_region_type, MODEL = model_type, TASK = HCP_task))

# Get participants don't have motor fMRI data 
subjects_missing_data = []
if HCP_task == "motor":
	sub_no_rl_motor_data = get_subjects_missing_3t_fmri_data(train_subs, "tfMRI_MOTOR_RL") 
	sub_no_lr_motor_data = get_subjects_missing_3t_fmri_data(train_subs, "tfMRI_MOTOR_LR") 
	subjects_missing_data = list(set(sub_no_lr_motor_data + sub_no_rl_motor_data))# in HCP 3T motor data

# One sided permutation test
pvals = np.zeros((len(train_subs) - len(subjects_missing_data), num_brain_regions))
for sub_idx, sub in enumerate(train_subs): 
	if sub not in subjects_missing_data:

		predictive_performance = np.load("../../data/encoding_models/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}.npy".format(SUB=sub, 
			REGION = brain_region_type, MODEL = model_type, TASK = HCP_task))
		null_distribution = load_dict("../../data/encoding_model_null_distributions/sub_{SUB}_corr_{REGION}_level_{MODEL}_{TASK}".format(SUB=sub, 
									REGION = brain_region_type, MODEL = model_type, TASK = HCP_task))
		pvals[sub_idx, :] = get_empirical_pvalue_one_sided_permutation_test(predictive_performance, null_distribution)

pvals_multiple_hypothesis_corrected = pval_multiple_hypothesis_benjamini_hochberg_fdr_correction(np.ravel(pvals), alpha) 

# Count for each brain region how many participants had predictive performances significantly higher than chance 
num_subjects_sig_predicted_per_brain_region = np.zeros(num_brain_regions)
for brain_region in np.arange(pvals_multiple_hypothesis_corrected.shape[1]):
     num_subjects_sig_predicted_per_brain_region[brain_region] = np.sum(pvals_multiple_hypothesis_corrected[:, brain_region] < alpha)

np.save("../../data/encoding_models/num_subjects_sig_predicted_per_brain_region_{REGION}_level_{MODEL}_{TASK}".format( REGION = brain_region_type, 
	MODEL = model_type, TASK = HCP_task), num_subjects_sig_predicted_per_brain_region)