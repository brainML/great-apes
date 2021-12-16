import numpy as np 
import pre_processing_utils as utils

clips = ["MOVIE1", "MOVIE2", "MOVIE3", "MOVIE4"]

#get TR by feature matrix
TR_by_Feature_CC1 = utils.mk_TR_by_Feature(clips[0], '../../data/7T_movie/WordNetFeatures.hdf5') # From HCP dataset
TR_by_Feature_HO1 = utils.mk_TR_by_Feature(clips[1], '../../data/7T_movie/WordNetFeatures.hdf5')
TR_by_Feature_CC2 = utils.mk_TR_by_Feature(clips[2], '../../data/7T_movie/WordNetFeatures.hdf5')
TR_by_Feature_HO2 = utils.mk_TR_by_Feature(clips[3], '../../data/7T_movie/WordNetFeatures.hdf5')

#Account for hemodynamic delay
TR_by_Feature_CC1_offset = utils.offset_feature_matrix_by_TRs(TR_by_Feature_CC1)
TR_by_Feature_HO1_offset = utils.offset_feature_matrix_by_TRs(TR_by_Feature_HO1)
TR_by_Feature_CC2_offset = utils.offset_feature_matrix_by_TRs(TR_by_Feature_CC2)
TR_by_Feature_HO2_offset = utils.offset_feature_matrix_by_TRs(TR_by_Feature_HO2)

# get TRs to drop 
CC1DropTRs_gallant = utils.get_visual_semantic_dropped_TRs("MOVIE1")
HO1DropTRs_gallant = utils.get_visual_semantic_dropped_TRs("MOVIE2")
CC2DropTRs_gallant = utils.get_visual_semantic_dropped_TRs("MOVIE3")
HO2DropTRs_gallant = utils.get_visual_semantic_dropped_TRs("MOVIE4")

# drop TRs with no visual semantic features 
CC1_TRsDropped_gallant = np.delete(TR_by_Feature_CC1_offset, CC1DropTRs_gallant, axis = 0)
HO1_TRsDropped_gallant = np.delete(TR_by_Feature_HO1_offset, HO1DropTRs_gallant, axis = 0)
CC2_TRsDropped_gallant = np.delete(TR_by_Feature_CC2_offset, CC2DropTRs_gallant, axis = 0)
HO2_TRsDropped_gallant = np.delete(TR_by_Feature_HO2_offset, HO2DropTRs_gallant, axis = 0)

# concatenate 4 movie runs 
all4MovieRuns_gallant_8TROffset = np.vstack((CC1_TRsDropped_gallant,
                                            HO1_TRsDropped_gallant, 
                            				CC2_TRsDropped_gallant, 
                                            HO2_TRsDropped_gallant))

np.save("../../data/movie_visual_semantic_features.npy", all4MovieRuns_gallant_8TROffset)