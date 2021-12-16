#!/bin/bash

mask_file=../../data/shen_1mm_268_parcellation_downsampled_to_1.6mm.nii.gz
participant_data=../../data/participant_data 
out_dir=${participant_data}/pre_processed_rest

for d in ${participant_data}/REST/*/
do
    sub=$(basename $d)
    if [ "$sub" != "$out_dir" ]
    then
        for run in rfMRI_REST1_7T_PA rfMRI_REST2_7T_AP rfMRI_REST3_7T_PA rfMRI_REST4_7T_AP 
        do
            3dROIstats -mask ${mask_file} -quiet ${d}/MNINonLinear/Results/${run}/${run}_hp2000_clean.nii.gz > ${out_dir}/${sub}_${run}_shen268_roi_ts.txt
        done
    fi
done
  
