import numpy as np
import cortex

# need subject and transform (xfmname) in pycortex filestore to run this code
# output files are available in ../../data/ to enable analyis without running this code

thin_mask_1point6mm = cortex.utils.get_cortical_mask("MNI", "atlas_1.6mm_from_fsl", type='thin')
np.save("../../data/thin_mask_1.6mm_MNI.npy", thin_mask_1point6mm)


thin_mask_2mm = cortex.utils.get_cortical_mask("MNI", "atlas_2mm", type='thin')
np.save("../../data/thin_mask_2mm_MNI.npy", thin_mask_2mm)

