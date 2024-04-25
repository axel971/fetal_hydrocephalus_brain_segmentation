import sys
sys.path.append('/home/axel/dev/fetal_hydrocephalus_segmentation/source')
import os
import tensorflow as tf


from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
import numpy as np
from model.image_process import crop_edge_pair, load_image_correct_oritation, crop3D_hotEncoding
from model.dataio import import_data_filename, write_nii
from model.one_hot_label import restore_labels
import time
import pickle
import gc

# from viewer import view3plane
# Just disables the warning and gpu info, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# check the gpu connection
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import pandas

from utils import get_nii_data, get_nii_affine, save_image
from imgaug import augmenters as iaa

def main():
    n_subject = 98
    subject_list = np.arange(n_subject)
    image_size = [117, 159, 126]
    patch_size = [64, 64, 64]
    labels = [0, 1]   # redefine labels
    # np.random.shuffle(subject_list)
    
    # Get path data
    file_dir = "/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.xlsx"
    
    data = pandas.read_excel(file_dir, index=False)
    data_length = data.shape
    subject_names = np.array(data['ID'])
    img_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/MRI/histogram_matching'
    MCsamples_dir = "/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/MCsamples"
    seg_dir = "/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/prediction_T0"
    
    affines = []
    
    for n in range(n_subject):
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '_preproc.nii'))


    
    for iSubject in range(n_subject):
    	
    	MCsamples = np.load(MCsamples_dir + '/' + subject_names[iSubject] + '_MCsamples.npy')
    	
    	# Compute the sum of the predicted probability maps
    	YProbaMap = np.sum(MCsamples, axis = 0)
#     	YProbaMap = MCsamples[5]
    	
    	YProbaMap = crop3D_hotEncoding(YProbaMap, image_size, len(labels))
    	Y = restore_labels(YProbaMap, labels)
    	
    	save_image(Y, affines[iSubject], seg_dir + "/" + subject_names[iSubject] + '_predicted_segmentation.nii')
#     	
    
if __name__ == '__main__':
    main()
