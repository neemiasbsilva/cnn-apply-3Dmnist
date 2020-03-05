import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from keras.layers import Activation, Input
from keras.regularizers import l2
import numpy as np
from keras.initializers import random_normal, constant


def fully_connected(x, nf, name, weight_decay):

    kerner_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Dense(nf, name=name, 
                kernel_regularizer=kerner_reg,
                bias_regulaizer=bias_reg,
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)

    return x


def conv3d(x, nf, ks, name, weight_decay):

    kernel = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv3D(nf, (ks, ks, ks), padding='same', name=name,
                kernel_regularizer=kernel,
                bias_regularizer=bias_reg,
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))

    return x

def relu(x): return Activation('relu')(x)

def pooling(x, ks, name):

    x = MaxPooling3D((ks, ks, ks), name=name)(x)

    return x

def softmax(x):
    return Activation('softmax')(x)

def get_train_model(sample_shape, dimension=3, layer_name='block1_conv3d', weight_decay=5e-4):

    inputs = []

    img_input = Input(shape=sample_shape)

    inputs.append(img_input)

    model = Sequential()

    model.add(Conv3D(32, (3, 3, 3), activation='relu', kernel_initalizer='he_uniform', input_shape=sample_shape))

    model = pooling(model, 2, 'MaxPooling3D_layer2')

    model = conv3d(model, 64, 3, 'Conv3d_layer3', weight_decay)

    model = relu(model)

    model = pooling(model, 2, 'MaxPooling3D_layer3')

    model.add(Flatten())

    model = fully_connected(model, 256, 'Dense_layer4', weight_decay)

    model = relu(model)

    model = pooling(model, 2, 'MaxPooling3D_layer5')

    model = fully_connected(model, 10)

    model = softmax(model)

    model = Model(img_input, model)

    return model