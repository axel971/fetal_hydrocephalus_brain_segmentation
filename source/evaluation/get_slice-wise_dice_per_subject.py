import sys
sys.path.append('/home/axel/dev/fetal_hydrocephalus_segmentation/source')

from utils import get_nii_data, get_nii_affine, save_image
import os
import numpy as np
from model.dataio import import_data_filename, write_nii
import metrics.metrics as metrics
import pandas as pd


n_subject = 98
subject_list = np.arange(n_subject)
labels = [0, 1]  # redefine labels
    
print('Process: slice-wise dice computation per subject')
    
# Get path data
file_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.xlsx'
data = pd.read_excel(file_dir)
data_length = data.shape
subject_names = np.array(data['ID'])

pred_delineation_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/prediction'
delineation_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/delineationn/relabeling'
dice_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/slice-wise_dice'

    
# Compute the Dice score per slice for each subject
for n in range(n_subject):

	dice = []
	delineation = get_nii_data(delineation_dir + '/' + subject_names[n] + '_relabeled_delineation.nii')
	pred_delineation = get_nii_data(pred_delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
	
	for i_slide in range(delineation.shape[2]):
		if(np.sum(delineation[:, :, i_slide]) > 0 and np.sum(pred_delineation[:, :, i_slide]) > 0):
			dice.append(metrics.dice_array(delineation[:, :, i_slide], pred_delineation[:, :, i_slide]))
		else:
			dice.append(1)
	
	print(subject_names[n])
	print(dice)
		
	np.save(dice_dir + '/' + subject_names[n] + '_slice-wise_dice.npy', dice)
    


    
