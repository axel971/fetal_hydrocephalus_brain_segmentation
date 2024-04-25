'''
3D patch tool box
201906 lizhao @ cnmc
'''
import numpy as np


def get_single_patch(X, start_pos, patch_size):
    "can be called without the patch pbject"
    Y = X[start_pos[0]:start_pos[0]+patch_size[0],
          start_pos[1]:start_pos[1]+patch_size[1],
          start_pos[2]:start_pos[2]+patch_size[2]]
    return Y


def put_single_patch(Y, X, start_pos, patch_size):
    "can be called without the patch pbject, for the put the batch label back in the prediction"
    Y[start_pos[0]:start_pos[0]+patch_size[0],
      start_pos[1]:start_pos[1]+patch_size[1],
      start_pos[2]:start_pos[2]+patch_size[2],
      :] += X
    return Y


class patch():
    def __init__(self, image_size, patch_size, stride=[8, 8, 8]):
        self.patch_size = patch_size
        self.stride = np.asarray(stride)
        self.image_size = np.asarray(image_size)
        self.n_patch_3direction = np.ceil((np.asarray(self.image_size)-self.patch_size)/self.stride).astype(int)+1
        self.n_patch = np.prod(self.n_patch_3direction)
        self.patch_indices = np.arange(self.n_patch)
        self.size_after_pad = (self.n_patch_3direction-1)*self.stride + self.patch_size
        self.pad_width = np.ceil((self.size_after_pad - self.image_size)/2).astype(int)

        start = [0, 0, 0]
        # stop = ((self.n_patch_3direction)*self.stride)
        # self.patch_location = np.asarray(np.mgrid[start[0]:stop[0]:self.stride[0],
        #                                           start[1]:stop[1]:self.stride[1],
        #                                           start[2]:stop[2]:self.stride[2]].reshape(3, -1).T, dtype=np.int)
        self.step = np.ceil(np.asarray(self.size_after_pad - self.patch_size)/self.stride).astype(int) + 1  # should be the same as n_patch_3direction
        self.patch_location = np.asarray(np.meshgrid(start[0]+np.arange(self.step[0])*self.stride[0],
                                                     start[1]+np.arange(self.step[1])*self.stride[1],
                                                     start[2]+np.arange(self.step[2])*self.stride[2])).reshape(3, -1).T.astype(np.int)
        self.patch_index = 0  # start from 0

    def __info__(self):
        print('patch size:' + str(self.patch_location.shape))
        print('pad_width:' + str(self.pad_width))
        print('n_patch_3direction:' + str(self.n_patch_3direction))
        print('step:' + str(self.step))
        print('current_index:' + str(self.patch_index))

    # def __patch_pad__(self, X, min_image=0):
    #     if np.any(self.pad_width > [0, 0, 0]):
    #         X = np.pad(X,
    #                    mode='constant',
    #                    constant_values=(min_image, min_image),
    #                    pad_width=((self.pad_width[0], self.pad_width[0]),
    #                               (self.pad_width[1], self.pad_width[1]),
    #                               (self.pad_width[2], self.pad_width[2]))
    #                    )
    #     return X

    def __get_single_patch__(self, X, patch_index):
        min_image = 0.0
        if np.any(self.pad_width > [0, 0, 0]):
            X = np.pad(X,
                       mode='constant',
                       constant_values=(min_image, min_image),
                       pad_width=((self.pad_width[0], self.pad_width[0]),
                                  (self.pad_width[1], self.pad_width[1]),
                                  (self.pad_width[2], self.pad_width[2]))
                       )
        start_pos = self.patch_location[patch_index]
        # print(start_pos)
        # print(self.patch_size)
        # print('X shape:'+str(X.shape))
        return X[start_pos[0]:start_pos[0]+self.patch_size[0],
                 start_pos[1]:start_pos[1]+self.patch_size[1],
                 start_pos[2]:start_pos[2]+self.patch_size[2]]
                 
    def __get_single_patch__without_padding_test__(self, X, patch_index):
 
        start_pos = self.patch_location[patch_index]
        # print(start_pos)
        # print(self.patch_size)
        # print('X shape:'+str(X.shape))
        return X[start_pos[0]:start_pos[0]+self.patch_size[0],
                 start_pos[1]:start_pos[1]+self.patch_size[1],
                 start_pos[2]:start_pos[2]+self.patch_size[2]]

    def __put_single_patch__(self, Y, X, patch_index):
        "for the put the batch label back in the prediction"
        start_pos = self.patch_location[patch_index]
        Y[start_pos[0]:start_pos[0]+self.patch_size[0],
          start_pos[1]:start_pos[1]+self.patch_size[1],
          start_pos[2]:start_pos[2]+self.patch_size[2],
          :] += X
        return Y
