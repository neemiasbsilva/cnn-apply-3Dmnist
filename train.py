from model import get_training_model
from preparation_data import get_dataset
from keras.optimizers import SGD
import argparse


parser = argparse.ArgumentParser(description='Training Three Dimension CNN')

parser.add_argument("-batch_size", action="store", required=True, help="batch size", type=int)
parser.add_argument("-no_epochs", action="store", required=True, help="Number of epochs", type=int)
parser.add_argument("-learning_rate", action="store", default=0.001, required=False, help="Learning Rate", type=float)
parser.add_argument("-validation_split", action="store", default=0.2, required=False, help="Validation Split", type=float)
parser.add_argument("-verbosity", action="store", default=1, required=False, help="Verbosity", type=float)
parser.add_argument("-path_data", action="store", required=True, help="The dataset of the 3Dmnist path", dest='experiment_name')


arguments = parser.parse_args()

batch_size = arguments.batch_size
no_epochs = arguments.no_epochs
lr = arguments.learning_rate
val_split = arguments.validation_split
verbosity = arguments.verbosity
path_data = arguments.path_data


x_train, y_train, x_test, y_test = get_dataset(path_data)

model = get_training_model(sample_shape=(16, 16, 16, 3))
