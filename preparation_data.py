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


# -- Process code --
# Load the HDF5 data file
with h5py.File("./full_dataset_vectors.h5", "r") as hf:

    # Split the data into training/test features/targets
    X_train = hf["X_train"][:]
    targets_train = hf["y_train"][:]
    X_test = hf["X_test"][:]
    targets_test = hf["y_test"][:]

    # Determine sample shape
    sample_shape = (16, 16, 16, 3)

    # Reshape data into 3D format
    X_train = rgb_data_transform(X_train)
    X_test = rgb_data_transform(X_test)

    # Convert target vectors to categorical targets
    targets_train = to_categorical(targets_train).astype(np.integer)
    targets_test = to_categorical(targets_test).astype(np.integer)
