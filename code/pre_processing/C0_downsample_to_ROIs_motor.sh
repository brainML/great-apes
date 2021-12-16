#!/bin/bash

mask_file=../../data/shen_2mm_268_parcellation.nii.gz
participant_data=../../data/participant_data
out_dir=${participant_data}/pre_processed_motor

for d in ${participant_data}/*/
do
    sub=${d%/}
    if [ "$sub" != "$out_dir" ]
    then 
        for motor_run in tfMRI_MOTOR_LR tfMRI_MOTOR_RL
        do
            3dROIstats -mask ${mask_file} -quiet  ${d}/MNINonLinear/Results/${motor_run}/${motor_run}.nii.gz > ${out_dir}/${sub}_${motor_run}_shen268_roi_ts.txt
        done
    fi
done
  
