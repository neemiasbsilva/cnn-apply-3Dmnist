from model import get_training_model
from preparation_data import get_dataset
from keras.optimizers import SGD
import argparse
import os

parser = argparse.ArgumentParser(description='Training Three Dimension CNN')

parser.add_argument("-batch_size", action="store", required=True, help="batch size", type=int)
parser.add_argument("-no_epochs", action="store", required=True, help="Number of epochs", type=int)
parser.add_argument("-learning_rate", action="store", default=0.001, required=False, help="Learning Rate", type=float)
parser.add_argument('-momentum', action="store", default=0.9, required=False, help="Momentum term", type=float)
parser.add_argument("-validation_split", action="store", default=0.2, required=False, help="Validation Split", type=float)
parser.add_argument("-verbosity", action="store", default=1, required=False, help="Verbosity", type=float)
parser.add_argument("-path_data", action="store", required=True, help="The dataset of the 3Dmnist path", dest='train_path')
parser.add_argument("-experiment_name", action="store", required=True, help="Folder to save the experiment", dest="experiment_name" )

arguments = parser.parse_args()

batch_size = arguments.batch_size
no_epochs = arguments.no_epochs
lr = arguments.learning_rate
momentum = arguments.momentum
val_split = arguments.validation_split
verbosity = arguments.verbosity
path_data = arguments.path_data
experiment_name = arguments.experiment_name

x_train, y_train, x_test, y_test = get_dataset(path_data)

model = get_training_model(sample_shape=(16, 16, 16, 3))

opt = SGD(lr=lr, momentum=momentum)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, y_train, 
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=val_split)

model.save(os.path.join(experiment_name, "model.h5"))