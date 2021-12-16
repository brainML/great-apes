input="../../data/MNI152_T1_1mm_brain"
flirt -in ${input} -ref ${input} -applyxfm -applyisoxfm 1.6 -nosearch -out ../../data/MNI125_T1_1mm_brain_downsampled_to_1.6mm
