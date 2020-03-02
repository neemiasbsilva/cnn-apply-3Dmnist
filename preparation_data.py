import h5py
import numpy as np

def rgb_data_transform(data):
    data_t = []

    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i])).reshape(16, 16, 16, 3)
    
    return np.asarray(data_t, dtype=np.float32)

def get_dataset(path):
    with h5py.File("path", 'r') as hf:
        
        # Split the data into training/test featrues/targets

        x_train = hf["X_train"][:]
        y_train = hf["y_train"][:]
        x_test = hf["X_test"][:]
        y_test = hf["y_test"][:]

        # Determine sample shape
        sample_shape(16, 16, 16, 3)

        # Reshape data into 3D format
        x_train = rgb_data_trainsfor(x_train)

