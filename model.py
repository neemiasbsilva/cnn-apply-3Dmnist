import keras
from keras import layers
from keras.model import Sequential, Model
from keras.layers import Dense, Flatten, Conv3d
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