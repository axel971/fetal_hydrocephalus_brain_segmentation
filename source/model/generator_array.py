
import numpy as np
import tensorflow.keras
from model.one_hot_label import multi_class_labels
from model.dataio import write_label_nii, write_nii
from model.patch3d import patch
from model.image_process import normlize_mean_std
from model.augmentation import create_affine_matrix, similarity_transform_volumes

import torchio as tio

class Generator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, X, Y, batch_size=32, patch_size=[64, 64, 64], labels=[1], stride=[1, 1, 1]):
        'Initialization'
        self.batch_size = batch_size
        self.patch_size = np.asarray(patch_size)

        self.labels = labels
        self.image_size = np.asarray(X.shape[1:])
        self.n_subject = X.shape[0]
        self.loc_patch = patch(np.asarray(X.shape[1:]), patch_size, stride)

        self.indices_images = np.array(range(self.n_subject))
        self.indices_patches = np.array(range(self.loc_patch.n_patch))
        np.random.shuffle(self.indices_images) # At each epoch the input data are suffle
        np.random.shuffle(self.indices_patches)
        
        if np.any(self.loc_patch.pad_width > [0, 0, 0]): # Pad X and Y if necessary
        	self.X = np.zeros(np.append(len(X),(self.loc_patch.size_after_pad)),  dtype=np.float32)
        	self.Y = np.zeros(np.append(len(Y),(self.loc_patch.size_after_pad)),  dtype=np.float32)
        	
        	for n in range(len(X)):
        		self.X[n] = np.pad(X[n], mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
        		self.Y[n] = np.pad(Y[n], mode='constant', constant_values=(0., 0.), pad_width=((self.loc_patch.pad_width[0], self.loc_patch.pad_width[0]), (self.loc_patch.pad_width[1], self.loc_patch.pad_width[1]), (self.loc_patch.pad_width[2], self.loc_patch.pad_width[2])))
        else:
        	self.X = X
        	self.Y = Y
        
        self.step_by_epoch = 400
        self.current_epoch = 0
        
        self.data_augmentation_transform = (tio.OneOf({tio.RandomAffine(scales = (1,1), degrees=(5,5,5), translation = (3, 3, 3)):0.5, tio.RandomFlip(axes = 0, flip_probability = 1): 0.5}, p = 0.75) )

    	

    def __len__(self):
        'Denotes the number of step per epoch'
        return self.step_by_epoch


    def __getitem__(self, index):
        'Generate one batch of data'

        # Built arrays to stock the delineations and images for one bacth
        X = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], len(self.labels)), dtype=np.float32)  # channel_last by default
          
        batch_image_indices = np.random.choice(self.indices_images, self.batch_size)
        batch_patch_indices = np.random.choice(self.indices_patches, self.batch_size)
        
        
        for batch_count in range(self.batch_size):
            image_index = batch_image_indices[batch_count]
            patch_index = batch_patch_indices[batch_count]

            label = self.loc_patch.__get_single_patch__without_padding_test__(self.Y[image_index], patch_index)
            image = self.loc_patch.__get_single_patch__without_padding_test__(self.X[image_index], patch_index)
            
            # Data augmentation
            patch_tio = tio.Subject(image = tio.ScalarImage(tensor = np.expand_dims(image, 0)), segmentation = tio.LabelMap(tensor = np.expand_dims(label, 0)))
            patch_tio_augmented = self.data_augmentation_transform(patch_tio)        
            
            X[batch_count, :, :, :, 0] = patch_tio_augmented['image'].numpy()[0]
            Y[batch_count] = multi_class_labels(patch_tio_augmented['segmentation'].numpy()[0], self.labels) #Perform hot encoding
            
        return X, Y
        
        

    def on_epoch_end(self):
        'Shuffle the image and patch indexes after each epoch'
        self.current_epoch += 1
        np.random.shuffle(self.indices_images)
        np.random.shuffle(self.indices_patches)
        

