import os
import numpy as np
import nibabel as nib


def get_nii_data(path):
	img = nib.load(path)
	data = img.get_fdata()
	
	if len(img.shape) == 4:

		return data[:,:,:,0]
	else:
		#print(data.shape)
		return data
	
def get_nii_affine(path):
	img = nib.load(path)
	affine = img.affine
	return affine
	
def get_voxel_spacing(path):
	img = nib.load(path)	
	return img.header.get_zooms()
	
def save_image(np_arr, affine, path):
	img = nib.Nifti1Image(np_arr, affine)
	nib.save(img, path)
	
def load_nii_image(paths):
	imgs = []
	affines = []
	for image_path in paths:
		imgs.append(get_nii_data(image_path))
		affines.append(get_nii_affine_image_path(image_path))
	return {
		'image': np.array(imgs),
		'affine': np.array(affines)
		}	
	