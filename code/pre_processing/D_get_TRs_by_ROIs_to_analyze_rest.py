import sys
sys.path.append("../") # allows python to look for modules in parent directory
from individual_differences_utils import save_dict
from pre_processing_utils import read_json_list, make_TR_by_ROI_dict
import scipy as sp
import numpy as np

# Get TR by ROI dictionary. The output is zscored 
input_kwargs ={
    "start_stop_pads": (10, 5), 
     subj_list="../../data/train_subjects_list.npy" } # Contact us for access to train_subjects_list 

subjwise_ts_dictM1 = make_TR_by_ROI_dict(clip="REST1",**input_kwargs)
subjwise_ts_dictM2 = make_TR_by_ROI_dict(clip="REST2",**input_kwargs) 
subjwise_ts_dictM3 = make_TR_by_ROI_dict(clip="REST3",**input_kwargs)
subjwise_ts_dictM4 = make_TR_by_ROI_dict(clip="REST4",**input_kwargs)

# Concatenate the 4 runs together per subject
train_subs = read_json_list(train_subjects_list) #contact us for access to train_subjects_list (the list of participants we have selected for the development set)
subjwise_ts_dict_4runs = {}
for subj in train_subs:
    subjM1 = subjwise_ts_dictM1[subj]
    subjM2 = subjwise_ts_dictM2[subj]
    subjM3 = subjwise_ts_dictM3[subj]
    subjM4 = subjwise_ts_dictM4[subj]
    subjwise_ts_dict_4runs[subj] = sp.stats.zscore(np.concatenate((np.concatenate((np.concatenate((subjM1, subjM2)), subjM3)), subjM4)))

save_dict(subjwise_ts_dict_4runs, "../../data/subjwise_dict_TR_by_ROI_matrix_zscoredPerRunAndThenAcross4Runs_rest")

