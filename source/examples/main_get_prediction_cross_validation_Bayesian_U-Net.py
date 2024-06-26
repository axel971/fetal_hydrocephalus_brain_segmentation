import sys
sys.path.append('/home/axel/dev/fetal_hydrocephalus_segmentation/source')

import os
import tensorflow as tf
from model.model import bayesian_Unet3d
from tensorflow.keras import optimizers
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator
import numpy as np
from model.prediction_my import BayesianPredict
from model.image_process import crop_edge_pair, load_image_correct_oritation
from model.dataio import import_data_filename, write_nii
from model.one_hot_label import redefine_label
import metrics.metrics as metrics
import time
import pickle
import gc
from sklearn.model_selection import KFold
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
    labels = [0, 1]  # redefine labels
    # np.random.shuffle(subject_list)
    
    # Get path data
    file_dir = "/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.xlsx"
    
    data = pandas.read_excel(file_dir)
    data_length = data.shape
    subject_names = np.array(data['ID'])
    img_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/MRI/histogram_matching'
    delineation_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/delineationn/relabeling'
    
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    Y = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype=np.float32)
    affines = []
    
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_preproc.nii' )
        Y[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_relabeled_delineation.nii')
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '_preproc.nii'))


    print('..data size:' + str(X.shape), flush=True)
    print('..loading data finished ' + str(time.asctime(time.localtime())), flush=True)
    
    
    nFold = 5
    foldSize = int(n_subject/nFold)
    
    dice = []  
    
    for iFold in range(nFold):
    	if iFold < nFold - 1 :
    		test_id = np.array(range(iFold * foldSize, (iFold + 1) * foldSize))
    		train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
    		train_id = train_id[0:((nFold - 1) * foldSize)]
    	else:
    		test_id = np.array(range(iFold * foldSize, n_subject)) # Manage the case where the number of subject is not a multiple of the number of fold
    		train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
    
    	
    	print('---------fold--'+str(iFold + 1)+'---------')
    	print('train: ', train_id)
    	print('test: ', test_id)

    	model = tf.keras.models.load_model('/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/model/model_' + str(iFold) + '.h5', custom_objects = {'dice_multi': metrics.dice_multi})    	
    	    	
    	predictor = BayesianPredict(model, image_size, patch_size, labels, T = 6)
    	
    	start_execution_time = time.time()
    	for i in range(len(test_id)):
    	
    		prediction, MCsamples = predictor.__run__(X[test_id[i]])
    		
    		dice.append(metrics.dice_multi_array(Y[test_id[i]], prediction, labels))
    		np.save('/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/MCsamples/' + subject_names[test_id[i]] + '_MCsamples.npy', MCsamples)
    		save_image(prediction, affines[test_id[i]], '/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/Bayesian_U-Net/p_01/prediction/' + subject_names[test_id[i]] + '_predicted_segmentation.nii')
    	
    	end_execution_time = time.time()
    	print('executation time:' + str((end_execution_time - start_execution_time)/len(test_id)))
 
  
    	del model
    	
    	K.clear_session()
    	gc.collect()

    print(np.mean(dice, 0))
    
if __name__ == '__main__':
    main()
