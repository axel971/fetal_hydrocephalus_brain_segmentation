# handing the data input and output
# Li Zhao @ 20181120

import os
import numpy as np  # linear algebra
import SimpleITK


def import_data_filename(data_dir, extension='.nii'):
    data = [x for x in sorted(os.listdir(data_dir)) if x.endswith(extension) and not x.startswith('.')]
    return [data_dir+tmp for tmp in data]


def load_single_image(filename):
    itk_obj = SimpleITK.ReadImage(filename)
    image = SimpleITK.GetArrayFromImage(itk_obj).astype(np.float32)
    return image


def write_segmentation_nii(data, filename, labels):
    image = np.zeros(data.shape[:-1], dtype=np.int16)
    for n in range(len(labels)):
        image[data[..., n] == 1] = labels[n]
    output = SimpleITK.GetImageFromArray(image)
    SimpleITK.WriteImage(output, filename)


def write_nii(data, filename='temp', direction=[1, 0, 0, 0, 1, 0, 0, 0, -1]):
    output = SimpleITK.GetImageFromArray(data)
    output.SetDirection(direction)
    SimpleITK.WriteImage(output, filename)
    # print(data.shape)


def write_label_nii(data, namekey):
    for n in range(data.shape[-1]):
        tmp = data[..., n]
        filename = namekey + str(n) + '.nii'
        write_nii(data=tmp, filename=filename)


def printgpu(text):
    print(text, flush=True)
