import sys
sys.path.append('/home/axel/dev/neonatal_brain_segmentation/source')

import numpy as np
import pandas
from utils import get_nii_data, get_nii_affine, save_image
from utils_uncertainty import normalize, min_max
from sklearn.metrics import recall_score, roc_auc_score, precision_score, auc

def main():

	n_subject = 36
	image_size = [256, 128, 256]

	# Get the name list of the subject
	list_subject_name_dir = "/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name_PHH.xlsx"
	data = pandas.read_excel(list_subject_name_dir, index=False)
	data_length = data.shape
	subject_names = np.array(data['subject_name'])
	
	# Get the ground truth segmentation
	gt_segmentation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/delineations/case/preprocessing/allVOIs'
	gt_segmentation = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	affines = []
	for n in range(n_subject):
		gt_segmentation[n] =  get_nii_data(gt_segmentation_dir + '/' + subject_names[n] + '_delineations.nii.gz')
		affines.append(get_nii_affine(gt_segmentation_dir + '/' + subject_names[n] + '_delineations.nii.gz'))
		gt_segmentation[gt_segmentation == 7] = 0 # Hack: the ground segmentations have one volume in add compared to the predicted segmentation	
		
	# Get the predicted segmentation
	predicted_segmentation_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_decoder/p_03/prediction/allVOI_cross_entropy_loss/'
	predicted_segmentation = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	for n in range(n_subject):
		predicted_segmentation[n] =  get_nii_data(predicted_segmentation_dir + '/' + subject_names[n] + '_predicted_segmentation.nii')
		
		
	# Compute the difference maps between the ground truth and the predicted segmentations
	difference_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
	for iSubject in range(0, n_subject):
		difference_maps[iSubject] = (gt_segmentation[iSubject] == predicted_segmentation[iSubject])
		save_image(difference_maps[iSubject], affines[iSubject], '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_decoder/p_03/difference_map/' + subject_names[iSubject] + '_difference_map.nii')
    	
	# Get the uncertainty maps
	uncertainty_dir = '/home/axel/dev/neonatal_brain_segmentation/data/output/bayesian_UNet/dropout_decoder/p_03/predictive_entropy/allVOI_cross_entropy_loss/'
	uncertainty_maps = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
	
	for n in range(n_subject):
		uncertainty_maps[n] =  get_nii_data(uncertainty_dir + '/' + subject_names[n] + '_predictiveEntropy.nii' )
		
	# Normalize the uncertainty maps by their maximum value inside the whole data set
	normalized_uncertainty_maps = normalize(uncertainty_maps)
	
	# Applied a threshold on the normalized uncertainty maps
	recall = np.zeros(10, dtype= np.float32)
	precision = np.zeros(10, dtype= np.float32)
  	
	for iThreshold in range(0,10):
		thresh_uncertainty_maps = (normalized_uncertainty_maps[1] < np.float(iThreshold/10))
		recall[iThreshold] = recall_score(difference_maps[1].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int), pos_label = 0)
		precision[iThreshold] = precision_score(difference_maps[1].flatten().astype(int), thresh_uncertainty_maps.flatten().astype(int), pos_label = 0)
	
	print(auc(recall, precision))
if __name__ == '__main__':
    main()
	