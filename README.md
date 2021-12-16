# great-apes

This repository contains the code to reproduce results in the manuscript:

"Behavior measures are predicted by how information is encoded in an individual’s brain"

by Jennifer Williams and Leila Wehbe at Carnegie Mellon University.


The code supports:
- Training ROI-level and voxel-wise encoding models, including Average Participant Encoding models (APEs)
- Relating individual differences in how information is encoded in a participant's brain to their behavior measures.
- The code is designed to work with the Human Connectome Project (HCP) fMRI data, but could be adapted for other datasets.

To reproduce the results in the manuscript, you will need to apply for access to the publicly available HCP data:
- Create a free account at https://db.humanconnectome.org.
- Read and accept the open access data use terms for the "WU-Minn HCP Data - 1200 Subjects" dataset. This will give you access to the fMRI data and most of the demographic and behavioral data. 
- Read the restricted data use terms and apply for access to this data under the "apply for restricted data" section at https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-restricted-data-use-terms. This will give you access to the family structure data, which is contained in the "Restricted" csv file.
- Contact us for access to the list of participants we have selected for the development set. 
- Use our pre-processing code (code/pre-processing) to pre-process the data. 
- Use our analysis code (code/analyses) to analyze the data.
