#!/bin/bash

mask_file=../../data/shen_1mm_268_parcellation_downsampled_to_mni_1.6mm.nii.gz
participant_data=../../data/participant_data
out_dir=${participant_data}/pre_processed_movie

for d in ${participant_data}/*/
do
    sub=${d%/}
    if [ "$sub" != "$out_dir" ]
    then
        for run in tfMRI_MOVIE1_7T_AP tfMRI_MOVIE2_7T_PA tfMRI_MOVIE3_7T_PA tfMRI_MOVIE4_7T_AP 
        do
            3dROIstats -mask ${mask_file} -quiet ${d}/MNINonLinear/Results/${run}/${run}_hp2000_clean.nii.gz > ${out_dir}/${sub}_${run}_shen268_roi_ts.txt
        done
    fi
done
  
