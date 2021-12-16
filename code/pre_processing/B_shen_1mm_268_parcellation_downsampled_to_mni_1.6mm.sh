input="../../data/shen_1mm_268_parcellation"
flirt -in ${input} -ref ../../data/MNI125_T1_1mm_brain_downsampled_to_1.6mm.nii.gz -applyxfm -usesqform -noresampblur -interp nearestneighbour -out ${input}_downsampled_to_mni_1.6mm
