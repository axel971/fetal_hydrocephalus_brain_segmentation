import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk


def crop_pad3D(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    small = 0
    y = np.ones(target_size, dtype=np.float32)*small
    current_size = x.shape
    pad_size = [0, 0, 0]

    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)
    start_pos = start_pos.astype(int)
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    return y

def crop3D_hotEncoding(x, target_size, nlabels):

    'Crop a hot encoding 3D image. The label dimension  is not crop'        
    output_size = np.append((target_size), nlabels)
    y = np.zeros(output_size, dtype=np.float32)

    start_pos = np.ceil((np.asarray(x.shape) - np.asarray(output_size))/2.0)
    start_pos = start_pos.astype(int)

    y = x[(start_pos[0]):(start_pos[0]+target_size[0]),
           (start_pos[1]):(start_pos[1]+target_size[1]),
           (start_pos[2]):(start_pos[2]+target_size[2]), ...]
           
    return y


def crop_edge3D(x, target_size):
    # print(x.shape)
    small = 0
    y = np.ones(target_size, dtype=np.float32)*small

    tmp = x - np.min(x)
    # print(np.sum(tmp, axis=(1,2)).shape)
    I0 = np.sum(tmp, axis=(1, 2))
    index0 = I0>0
    I1 = np.sum(tmp, axis=(0, 2))
    index1 = I1>0
    I2 = np.sum(tmp, axis=(0, 1))
    index2 = I2>0
    new_shape = (sum(index0), sum(index1), sum(index2))
    image_cropped = x[index0, ...]
    image_cropped = image_cropped[:, index1, :]
    image_cropped = image_cropped[..., index2]
    # print(new_shape)
    # print('crop image: '+str(image_cropped.shape))

    y = zoom(image_cropped, (target_size[0]/len(index0), target_size[1]/len(index1), target_size[2]/len(index2)))
    return y


def resize(x, target_size, islabel=False):
    current_size = x.shape
    if islabel:
        y = np.round(zoom(x, np.divide(target_size, current_size), order=0))
    else:
        y = zoom(x, np.divide(target_size, current_size), order=1)
    return y


def crop_edge_pair(path, target_size):
    # print(x.shape)
    small = 0
    y = np.ones(target_size, dtype=np.float32)*small

    x0 = load_image_correct_oritation(path)    #  image
    y0 = load_image_correct_oritation((path[:-14]+'9tissue_labels.nii.gz'))  # label

    # crop based on the labels
    tmp = y0 - np.min(y0)
    # print(np.sum(tmp, axis=(1,2)).shape)
    I0 = np.sum(tmp, axis=(1, 2))
    index0 = I0>0
    I1 = np.sum(tmp, axis=(0, 2))
    index1 = I1>0
    I2 = np.sum(tmp, axis=(0, 1))
    index2 = I2>0

    new_shape = (sum(index0), sum(index1), sum(index2))

    y_cropped = y0[index0, ...]
    y_cropped = y_cropped[:, index1, :]
    y_cropped = y_cropped[..., index2]
    x_cropped = x0[index0, ...]
    x_cropped = x_cropped[:, index1, :]
    x_cropped = x_cropped[..., index2]
    # print(new_shape)
    # print('crop image: '+str(image_cropped.shape))
    # print(np.divide(target_size, new_shape))
    x = zoom(x_cropped, np.divide(target_size, new_shape), order=1)
    y = np.around(zoom(y_cropped, np.divide(target_size, new_shape), order=0))
    return x, y


def load_image_correct_oritation(filename):
    image_obj = sitk.ReadImage(filename)
    direction = image_obj.GetDirection()
    origin = np.asarray(image_obj.GetOrigin())
    spacing = np.asarray(image_obj.GetSpacing())
    affine = SimpleRot(direction)
    data = sitk.GetArrayFromImage(image_obj).astype(np.float32)
    image_size = np.asarray(data.shape)
    center = (image_size/2-1)*spacing
    affine.SetCenter([center[0], center[1], center[2]])
    image_obj.SetOrigin([0,0,0])
    image_obj.SetDirection([1,0,0,0,1,0,0,0,1])
    newimage = resample(image_obj, affine)
    data = sitk.GetArrayFromImage(newimage).astype(np.float32)
    return data


def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 0
    # print(transform)
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def SimpleRot(matrix):
    dimension = 3
    affine = sitk.AffineTransform(3)
    matrix = np.array(matrix).reshape((dimension, dimension))
    target = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,-1]])
    transform_matrix = target @ np.linalg.inv(matrix)
    affine.SetMatrix(transform_matrix.ravel())
    return affine


def normlize_mean_std(tmp):
    tmp_std = np.std(tmp) + 0.0001
    tmp_mean = np.mean(tmp)
    tmp = (tmp - tmp_mean) / tmp_std
    return tmp


def normlize_min_max(tmp):
    tmp_max = np.amax(tmp)
    tmp_min = np.amin(tmp)
    tmp = (tmp - tmp_min) / (tmp_max-tmp_min)
    return tmp
