import keras
from keras import layers
from keras.model import Sequential, Model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from keras.layers import Activation, Input
from keras.regularizers import l2
import numpy as np


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
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                kernel_initializer=random_norlma(stddev=0.01),
                bias_initializer=constant(0.0))

    return x

def relu(x): return Activation('relu')(x)

def pooling(x, ks, name):

    x = MaxPooling3D((ks, ks, ks), name=name)(x)

    return x