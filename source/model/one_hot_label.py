import numpy as np
from tensorflow.keras.utils import to_categorical


def redefine_label(data):
    'manual redefine the labels'
    y = np.ones(data.shape, np.float32) * 6  # others labeld as 6

    y[data == 0] = 0  # 0 background
    y[np.logical_or((data == 1), (data == 5))] = 1  # 1 CSF + laterial ventricles
    y[data == 2] = 2  # 2 GM
    y[data == 3] = 3  # 3 WM
    y[np.logical_or((data == 7), (data == 9))] = 4  # 4 deep grey matter
    y[data == 6] = 5  # 5 cerebellum
    # y[np.logical_not(y)] = 6  # 6 others

    return y


def multi_class_labels(data, labels=[1]):
    n_labels = len(labels)
    new_shape = np.append(data.shape, n_labels)
    y = np.zeros(new_shape, np.float32)
    for label_index in range(n_labels):
        # y[..., label_index][data == labels[label_index]] = 1
        y[data == labels[label_index], label_index] = 1
    # y[data == 8, 2] = 1.0
    # y[data == 1, 1] = 1.0
    # y[np.logical_and(data != 1, data != 8), 0] = 1.0
    return y


def restore_labels(x, labels):
    tmp = np.squeeze(np.argmax(x, -1)).astype(np.int8)
    y = np.zeros(tmp.shape, np.int8)
    n_labels = len(labels)
    for label_index in range(n_labels):
        y[tmp == label_index] = labels[label_index]
    return y


def to_hot_label(X, labels, dtype=np.float32):
    input_shape = X.shape
    n_pixel = np.prod(input_shape)
    if not labels:
        labels = range(np.amax(X) + 1)  # suppose the label alway start from 0
    else:  # re-label the X
        for n, label_name in enumerate(labels):
            X[X<=label_name] = n
    n_label = len(labels)

    categorical = np.zeros((n_pixel, n_label), dtype=dtype)
    categorical[range(n_pixel), X] = 1

    output_shape = input_shape + (n_label)
    categorical = np.reshape(categorical, output_shape)
    return categorical
