import nibabel as nib
import nilearn.image as nil_image
from scipy.ndimage import zoom, affine_transform
import numpy as np
import SimpleITK as sitk

def create_rotation_matrix(param):
    '''
    Create a rotation matrix from 3 rotation angels around X, Y, and Z:
    =================
    Arguments:
        param: numpy 1*3 array for [x, y, z] angels in degree.

    Output:
        rot: Correspond 3*3 rotation matrix rotated around y->x->z axises.
    '''
    theta_x = param[0] * np.pi / 180
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = param[1] * np.pi / 180
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    theta_z = param[2] * np.pi / 180
    cz = np.cos(theta_z)
    sz = np.sin(theta_z)

    Rx = [[1, 0, 0],
          [0, cx, -sx],
          [0, sx, cx]]

    Ry = [[cy, 0, sy],
          [0, 1, 0],
          [-sy, 0, cy]]

    Rz = [[cz, -sz, 0],
          [sz, cz, 0],
          [0, 0, 1]]

    # Apply the rotation first around Y then X then Z.
    # To follow ITK transformation functions.
    rot = np.matmul(Rz, Ry)
    rot = np.matmul(rot, Rx)

    return rot

def create_affine_matrix(
    scale,
    rotation,
    translation,
    image_size,
):
        scale = np.random.uniform(scale[0], scale[1])

        if np.size(rotation) == 2:
            rotation = np.random.uniform(rotation[0], rotation[1], 3).astype(np.int32)
            # Create rotation Matrix
            rot = create_rotation_matrix(rotation.astype(np.int32))
        else:
            rot = rotation

        affine_trans_rot = np.eye(4)
        affine_trans_rot[:3, :3] = rot


        translation = np.random.uniform(translation[0], translation[1], 3)

        # Create scale matrix
        affine_trans_scale = np.diag([scale, scale, scale, 1.])
        # Create translation matrix
        affine_trans_translation = np.eye(4)
        affine_trans_translation[:, 3] = [translation[0],
                                          translation[1],
                                          translation[2],
                                          1]

        # Create shift & unshift matrix to apply rotation around
        # center of image not (0,0,0)
        shift = - np.asarray(image_size) // 2
        affine_trans_shift = np.eye(4)
        affine_trans_shift[:, 3] = [shift[0],
                                    shift[1],
                                    shift[2],
                                    1]

        unshift = - shift
        affine_trans_unshift = np.eye(4)
        affine_trans_unshift[:, 3] = [unshift[0],
                                      unshift[1],
                                      unshift[2],
                                      1]

        # Apply transformations
        affine_trans = np.matmul(affine_trans_scale, affine_trans_translation)
        affine_trans = np.matmul(affine_trans, affine_trans_unshift)
        affine_trans = np.matmul(affine_trans, affine_trans_rot)
        affine_trans = np.matmul(affine_trans, affine_trans_shift)
        return affine_trans, rotation
    
    
def similarity_transform_volumes(
    image,
    affine_trans,
    target_size,
    interpolation = 'nearest',
):
    image_size = np.shape(image)
    possible_scales = np.true_divide(image_size, target_size)
    crop_scale = np.max(possible_scales)
    if crop_scale <= 1:
        crop_scale = 1
    scale_transform = np.diag((crop_scale,
                               crop_scale,
                               crop_scale,
                               1))
    shift = -(
        np.asarray(target_size) - np.asarray(
            image_size // np.asarray(crop_scale),
        )
    ) // 2
    affine_trans_to_center = np.eye(4)
    affine_trans_to_center[:, 3] = [shift[0],
                                    shift[1],
                                    shift[2],
                                    1]

    transform = np.matmul(affine_trans, scale_transform)
    transform = np.matmul(transform, affine_trans_to_center)
    # #
    nifti_img = nib.Nifti1Image(image, affine=np.eye(4))
    nifti_image_t = nil_image.resample_img(
        nifti_img,
        target_affine=transform,
        target_shape=target_size,
        interpolation=interpolation,
    )
    image_t = nifti_image_t.get_data()
    # image_t = affine_transform(image, affine_trans)

    return image_t, transform


# affine, rotation = create_affine_matrix([1,1], [-175,175], [0,0], (64,64,64))
# 
# image, transform = similarity_transform_volumes(msk, affine, (64,64,64))
