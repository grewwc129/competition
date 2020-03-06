from keras.models import *
from keras.layers import *
from keras.regularizers import l2
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import keras.backend as K


def dense(input_shape=(50, 1), lr=1e-4):
    l2_regularize = 1e-3
    units = [64, 32, 16, 8]

    inputs = Input(shape=input_shape)
    x = inputs
    for unit in units:
        x = Dense(unit, activation='relu',
                  kernel_regularizer=l2(l2_regularize))(x)
        x = Dropout(0.25)(x)

    x = Dense(3, activation='softmax',
              kernel_regularizer=l2(l2_regularize))(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=lr),
                  losses=sparse_categorical_crossentropy, metrics=['acc'])
    return model
