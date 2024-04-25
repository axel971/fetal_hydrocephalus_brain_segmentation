import numpy as np
from scipy.ndimage import zoom
from dataio import write_nii
from image_process import load_image_correct_oritation
from one_hot_label import redefine_label


def main():
    input_folder = 'rawdata/'
    image_size = [80, 110, 90]
    is_label = [False, True, True]
    for n in range(46):
        data_path = []
        data_path.append(input_folder + str(n) + 'x.nii.gz')  # 0 is the reference
        data_path.append(input_folder + str(n) + 'y.nii.gz')
        data_path.append('drawem/output/' + str(n) + '.nii.gz')
        X = crop_edge_pair(data_path, is_label, image_size)
        # write_nii(X[0], 'data/' + str(n) + 'x.nii')
        # tmp = redefine_label(X[1])
        # write_nii(tmp, 'data/' + str(n) + 'y.nii')
        tmp = redefine_label_drawem(X[2])
        write_nii(tmp, 'data/' + str(n) + 'd.nii')


def redefine_label_drawem(data):
    'manual redefine the labels'
    y = np.ones(data.shape, np.float32) * 6  # others labeld as 6

    y[np.logical_or((data == 0), (data == 4))] = 0  # 0 background+4 non brain
    y[np.logical_or((data == 1), (data == 5))] = 1  # 1 CSF+laterial ventricles
    y[data == 2] = 2  # 2 GM
    y[data == 3] = 3  # 3 WM
    y[np.logical_or((data == 7), (data == 9))] = 4  # 4 deep grey matter
    y[data == 6] = 5  # 5 cerebellum
    # y[np.logical_not(y)] = 6  # 6 others

    return y


def crop_edge_pair(data_path, is_label, target_size):
    n_data = len(data_path)

    x = []
    y = []
    for n in range(n_data):
        x.append(load_image_correct_oritation(data_path[n]))    # image

    # crop based on the labels
    # tmp = x[0] - np.min(x[0])
    tmp = x[0] > np.std(x[0])
    I0 = np.sum(tmp, axis=(1, 2))
    index0 = I0 > 0
    I1 = np.sum(tmp, axis=(0, 2))
    index1 = I1 > 0
    I2 = np.sum(tmp, axis=(0, 1))
    index2 = I2 > 0

    new_shape = (sum(index0), sum(index1), sum(index2))

    for n in range(n_data):
        x_cropped = x[n][index0, ...]
        x_cropped = x_cropped[:, index1, :]
        x_cropped = x_cropped[..., index2]
        if is_label[n]:
            y.append(np.around(zoom(x_cropped,
                                    np.divide(target_size, new_shape),
                                    order=0)))
        else:
            y.append(zoom(x_cropped,
                          np.divide(target_size, new_shape),
                          order=1))
    return y


if __name__ == '__main__':
    main()
