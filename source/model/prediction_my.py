import sys
sys.path.append('/home/axel/dev/fetal_hydrocephalus_segmentation/source')

import numpy as np
from model.patch3d import patch
from model.image_process import normlize_mean_std, crop_pad3D, crop3D_hotEncoding
import metrics.metrics as metrics
from model.dataio import write_nii
from model.one_hot_label import restore_labels
import time

class predict(object):
    'run the model on test data'

    def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1]):
        'Initialization'
        self.patch_size = patch_size
        self.labels = labels
        self.stride = [14, 14, 14]
        self.image_size = np.asarray(image_size)
        self.model = model
        self.loc_patch = patch(self.image_size, patch_size, self.stride)
        self.batch_size = 6
        
        
    def __run__(self, X):
    	'test on one image each time'
    	
    	# Pad the input image X if necessary
    	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
    		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
    	else:
    		X_pad = X
    	   	
#     	output_size = np.append((self.loc_patch.size_after_pad),len(self.labels))
    	Y0 = np.zeros(np.append((self.loc_patch.size_after_pad),len(self.labels)),  dtype=np.float32) #Initialization: padded uncertainty map array
    	X0 = np.zeros([self.batch_size]+self.patch_size+[1]) #Initialization: patches used as batch 
    	
    	const_array = np.asarray(range(self.batch_size)) #Initialize: array with elements are iteratively equal to 0 until batch_size-1
    	 
    	for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
    		batch_of_patch_index =  const_array + self.batch_size*index
    		batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    		# get one batch_size
    		for n, selected_patch in enumerate(batch_of_patch_index):
    			X0[n, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, selected_patch)
    			
    		prediction = self.model(X0, training = False)

    		for n, selected_patch in enumerate(batch_of_patch_index):
    			Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction[n]), selected_patch)
    		
    	Y0 = crop3D_hotEncoding(Y0, self.image_size, len(self.labels))	 
    	
    	return restore_labels(Y0, self.labels) #return Y0 without hot encoding
    	

class BayesianPredict(object):
    'run the model on test data'

    def __init__(self, model, image_size, patch_size=(32, 32, 32), labels=[1], T = 6):
        'Initialization'
        self.patch_size = patch_size
        self.labels = labels
        self.stride = [14, 14, 14]
        self.image_size = np.asarray(image_size)
        self.model = model
        self.loc_patch = patch(self.image_size, patch_size, self.stride)
        self.T = T
        self.batch_size = 6

    def __run__(self, X):
    	'test on one image each time'
    
		# Pad the input image X if necessary
    	if np.any(self.loc_patch.pad_width > [0, 0, 0]):
    		X_pad = np.pad(X, mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
    	else:
    		X_pad = X
    	   	
    	   	
    	output_size = np.append((self.loc_patch.size_after_pad), len(self.labels))
    	output_size = np.append(self.T, output_size)
    	
    	Y0 = np.zeros(output_size,  dtype=np.float32)
    	X0 = np.zeros([self.batch_size]+self.patch_size+[1])
    	
    	const_array = np.asarray(range(self.batch_size)) #Initialize: array with elements are iteratively equal to 0 until batch_size-1
    	
    			  			  	
    	for iPass in range(self.T):
    	
    		for index in range(np.ceil(self.loc_patch.n_patch/self.batch_size).astype(int)):
    			batch_of_patch_index = const_array + self.batch_size*index
    			batch_of_patch_index[batch_of_patch_index>=self.loc_patch.n_patch] = 0 # Is this sentence is usefull ? (check with Li)
    		
    			# get one batch_size
    			for n, selected_patch in enumerate(batch_of_patch_index):
    				X0[n, :,:,:,0] = self.loc_patch.__get_single_patch__without_padding_test__(X_pad, selected_patch)
    			
    			# predict the segmentation patches for the current batch
    			prediction = self.model(X0, training = False)
   		
    			# put the label back
    			for n, selected_patch in enumerate(batch_of_patch_index):
    				Y0[iPass] = self.loc_patch.__put_single_patch__(Y0[iPass], np.squeeze(prediction[n]), selected_patch) #Put single patch perform patch-wise addition
    		
    	
    	# Compute the sum of the predicted probability maps
    	YProbaMap = np.sum(Y0, axis = 0)
    	
    	YProbaMap = crop3D_hotEncoding(YProbaMap, self.image_size, len(self.labels))
    	Y = restore_labels(YProbaMap, self.labels)
    	
    	return Y, Y0
    	

def test(x, model, image_size, patch_size, labels):
    prediction = np.zeros(x.shape)
    predictor = predict(model, image_size, patch_size, labels)
    for n in range(x.shape[0]):
        prediction[n] = predictor.__run__(x[n])
    return prediction


def bayesian_test(x, model, image_size, patch_size, labels):

    predictions = np.zeros(x.shape)
    probaMaps = np.zeros(np.append((x.shape), len(labels)))
    
    predictor = BayesianPredict(model, image_size, patch_size, labels)
    
    for n in range(x.shape[0]):
        predictions[n], probaMaps[n] = predictor.__run__(x[n])
        
    return predictions, probaMaps
    
def evaluate(x, y_true, model, image_size, patch_size, labels, ID, output_path='/home/axel/dev/neonatal_brain_segmentation/source/output/'):
    if not output_path:
        output_path = 'output/'
    n_subject = x.shape[0]
    predictor = predict(model, image_size, patch_size, labels)
    metric = 0.0
    np.set_printoptions(precision=3)
    for n in range(n_subject):
        y_pred = predictor.__run__(x[n])
        tmp = metrics.dice_multi_array(y_true[n], y_pred, labels)
        print(str(n)+': '+str(tmp))
        metric += tmp
        write_nii(y_pred, output_path+str(ID[n])+'test.nii')
    return metric/n_subject
