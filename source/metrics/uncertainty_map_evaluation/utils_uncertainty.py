import numpy as np

def min_max(arrays):
 
	min = np.amin(arrays)
	max = np.amax(arrays)
	
	return min, max

def normalize(arrays):

	min, max = min_max(arrays)
	arrays  = (arrays - min) / np.abs(max - min)
	
	return arrays
	