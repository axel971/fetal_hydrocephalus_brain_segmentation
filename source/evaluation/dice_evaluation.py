import sys
sys.path.append('/home/axel/dev/fetal_hydrocephalus_segmentation/source')

from utils import get_nii_data, get_nii_affine, save_image
import os
import numpy as np
from model.dataio import import_data_filename, write_nii
import metrics.metrics as metrics
import time
import pickle
import gc
import pandas as pd


n_subject = 98
subject_list = np.arange(n_subject)
labels = [0, 1]  # redefine labels
    
print('Process: Dice computation')
    
# Get path data
file_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.xlsx'
data = pd.read_excel(file_dir)
data_length = data.shape
subject_names = np.array(data['ID'])

# pred_delineation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/U-Net/experiment1/prediction/allVOI_crossEntropy'
pred_delineation_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/prediction_T0'
delineation_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/delineationn/relabeling'
dice_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/dice/dice_T0.csv'

    
# Compute the Dice score for each subject
dice = []
    
for n in range(n_subject):
    dice.append(metrics.dice_multi_array(get_nii_data(delineation_dir + '/' + subject_names[n] + '_relabeled_delineation.nii'), get_nii_data(pred_delineation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii'), labels))
	
data = pd.DataFrame(dice)
data.to_csv(dice_dir)
    


    
