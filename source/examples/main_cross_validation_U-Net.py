import sys
sys.path.append('/home/axel/dev/fetal_hydrocephalus_segmentation/source')
import os
from model.model import unet3d
from tensorflow.keras import optimizers
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator
import numpy as np
from model.prediction_my import evaluate, test
from model.image_process import crop_edge_pair, load_image_correct_oritation
from model.dataio import import_data_filename, write_nii
from model.one_hot_label import redefine_label
import metrics.metrics as metrics
import time
import pickle
import gc
# from sklearn.model_selection import KFold
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

	# Lood the data
    n_subject = 98
    subject_list = np.arange(n_subject)
    image_size = [117, 159, 126]
    patch_size = [64, 64, 64]
    labels = [0, 1]  # redefine labels
    
    file_dir = '/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.xlsx'
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
        Y[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_relabeled_delineation.nii' )
        affines.append(get_nii_affine(img_dir + '/' + subject_names[n] + '_preproc.nii'))

    print('..data size:' + str(X.shape), flush=True)
    print('..loading data finished')
    
	# Cross-validation process ...
    nFold = 5
    foldSize = int(n_subject/nFold)
    
    dice = []  
    
    for iFold in range(nFold):
    	
    	# Split the whole cohort on training and validation data sets for the current fold
    	if iFold < nFold - 1 :
    		test_id = np.array(range(iFold * foldSize, (iFold + 1) * foldSize))
    		train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
    		train_id = train_id[0:((nFold - 1) * foldSize)]
    	else:
    		test_id = np.array(range(iFold * foldSize, n_subject)) # Manage the case where the number of subject is not a multiple of the number of fold
    		train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
 
    	
    	print('---------fold--'+str(iFold + 1)+'---------')
    	print('training set: ', train_id)
    	print('training set size: ', len(train_id))
    	print('testing set: ', test_id)
    	print('testing set size: ', len(test_id))
    	

    	model = unet3d(patch_size+[1], len(labels))
    		
    	optimizer = optimizers.Adam(learning_rate=1e-3)
    	

		# For training on complete cohort, 
    	#training_generator = Generator(X, Y, batch_size = 4, patch_size=patch_size, labels=labels)
		# For training vs prediction on selected cohort, 
    	training_generator = Generator(X[train_id, ...], Y[train_id, ...], batch_size = 4, patch_size=patch_size, labels=labels)
    	
    	print('..generator_len: ' + str(training_generator.__len__()), flush=True)
    	print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
    	print('..training_n_patch_per_sub: ' + str(training_generator.loc_patch.n_patch), flush=True)

    	model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=[metrics.dice_for_iVOI])
    	
    	start_training_time = time.time()	
    	model.fit(x=training_generator, epochs=100, verbose=2)
    	end_training_time = time.time()	
    	print('training time: ' + str(end_training_time - start_training_time))

    	model.save('/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/U-Net/model/model_' + str(iFold) + '.h5')

    	##### Prediction ####
    	
    	start_execution_time = time.time()
    	prediction = test(X[test_id], model, image_size, patch_size, labels)
    	end_execution_time = time.time()
    	print('executation time:' + str((end_execution_time - start_execution_time)/len(test_id)))
    	
    	for i in range(len(prediction)):
    		dice.append(metrics.dice_multi_array(Y[test_id[i]], prediction[i], labels))
    		save_image(prediction[i], affines[test_id[i]], '/home/axel/dev/fetal_hydrocephalus_segmentation/data/output/U-Net/prediction/' + subject_names[test_id[i]] + '_predicted_segmentation.nii')
    		

    	del model
	    	
    	del training_generator
    	K.clear_session()
    	gc.collect()

    print(np.mean(dice, 0))
    
if __name__ == '__main__':
    main()
