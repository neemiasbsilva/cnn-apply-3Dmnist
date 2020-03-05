import h5py
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt 


def array_to_color(arr, cmap="Oranges"):
    
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    
    return s_m.to_rgba(arr)[:,:-1]


def rgb_data_transform(data):
    data_t = []

    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i])).reshape(16, 16, 16, 3)
    
    return np.asarray(data_t, dtype=np.float32)


def get_dataset(path):
    with h5py.File(path, 'r') as hf:
        
        # Split the data into training/test featrues/targets

        x_train = hf["X_train"][:]
        y_train = hf["y_train"][:]
        x_test = hf["X_test"][:]
        y_test = hf["y_test"][:]

        # Determine sample shape
        sample_shape = (16, 16, 16, 3)

        # Reshape data into 3D format
        x_train = rgb_data_transform(x_train)
        x_test = rgb_data_transform(x_test)

        # Convert target vectors to categorical targets
        y_train = to_categorical(y_train).astype(np.integer)
        y_test = to_categorical(y_test).astype(np.integer)

        return x_train, y_train, x_test, y_test

